import copy
import math

import torch.nn.functional as F
from torch import nn
import torch
from math import sqrt
from models_others import SoftAttn


class AWFLN(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(AWFLN, self).__init__()
        self.blk_5_30_3 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(5, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blk_4_30_3 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(4, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blk2 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )
        ##### up sampling #############
        self.up_sample4 = UpsampleBLock(4, 4)  ### for 4 band 4, for 8 band 8 ,in_channels, up_scale
        self.up_sample2 = UpsampleBLock(4, 2)  ### for 4 band 4, for 8 band 8

        self.conv6 = nn.Conv2d(in_channels=30, out_channels=4, kernel_size=3, stride=1, padding=1,
                               bias=True)  # change out as 4   or   8

        self.lu_block1 = RMRS(60)
        self.lu_block2 = RMRS(30)

    #######################
    def forward(self, ms_up, ms_org, pan):

        ms_org_up = self.up_sample4(ms_org)  ## in_channels, in_channels * up_scale ** 2

        data1 = torch.cat([ms_org_up, pan], dim=1)
        # ms_up4 = self.up_sample2(ms_up2)
        pan_conv = self.blk_5_30_3(data1)
        ms_up_conv = self.blk_4_30_3(ms_up)
        # pand_d = self.downscale(pan)

        out1 = torch.cat([ms_up_conv,  pan_conv], dim=1)

        out2 = self.lu_block1(out1)

        out3 = self.blk2(out2)
        out3 = self.lu_block2(out3)

        out8 = self.conv6(out3)
        out_f = out8 + ms_up

        return out_f


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()  # 负数部分的参数会变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=5, last=nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        elif kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out


# 通道注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction_ratio), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class RMRS(nn.Module):
    def __init__(self, out_channels):
        super(RMRS, self).__init__()

        self.conv2_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=2, dilation=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=3, dilation=3, bias=True)


        self.conv3_1 = AFLB(out_channels, out_channels, 3, 1, 1, use_bias=True)
        self.conv3_2 = AFLB(out_channels, out_channels, 3, 1, 1, use_bias=True)

        self.relu = nn.ReLU(inplace=True)
        init_weights(self.conv2_1, self.conv2_2, self.conv2_3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):

        out1 = x
        out21 = self.conv2_1(out1)
        out22 = self.conv2_2(out1)
        out23 = self.conv2_3(out1)

        out2 = torch.cat([out21, out22, out23], 1)
        out2 = self.conv3_1(out2)
        out2 = self.conv3_2(out2)

        out2 = self.relu(torch.add(out2, out1))
        return out2

class AFLB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=False):
        super(AFLB, self).__init__()
        self.in_features = in_planes
        self.out_features = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        # self.ch_att = ChannelAttention(kernel_size ** 2)  # change

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            SoftAttn(kernel_size ** 2)
        )

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        o_f = self.out_features
        k_size = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k_size) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k_size) / self.stride)
        att1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        att1 = att1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        att1 = att1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        att1 = att1.view(b, n_H, n_W, n * k_size * k_size)  # b,n_H,n_W,n*k*k

        att2 = att1  # *att2 #b,n_H,n_W,n*k*k
        att2 = att2.view(b, n_H * n_W, n * k_size * k_size)  # b,n_H*n_W,n*k*k
        att2 = att2.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx_unf = F.unfold(x, kernel_size=k_size, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = att2 * kx_unf  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k_size * k_size)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(o_f, n * k_size * k_size)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        out = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        out = out.view(b, n_H * n_W, o_f)  # b,n_H*n_W,m

        out = out.permute([0, 2, 1])  # b,m,n_H*n_W
        out = F.fold(out, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return out