import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_set_py.data_utils_RS import TestDatasetFromFolder
# from data import get_test_set
from model_4b import Generator  ################ need to change ######################
# from model_8_1 import Generator  ################ need to change ######################
# sys.path.append('E:\\remote sense image fusion\\shared_py')
from data_set_py.save_image_func import save_image_RS
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(test_data_loader, model, sate):
    val_bar = tqdm(test_data_loader)  # 验证集的进度条
    count = 0
    for ms_up_crop, ms_org_crop, pan_crop, gt_crop in val_bar:
        batch_size = ms_up_crop.size(0)  # current batch size
        # valing_results['batch_sizes'] += batch_size
        # detail_crop = detail_crop.type(torch.FloatTensor)  # to make the type the same as model
        # data = torch.cat(ms_up_crop, pan_crop)

        with torch.no_grad():  # validation
            ms_up = Variable(ms_up_crop)
            ms = Variable(ms_org_crop)
            pan = Variable(pan_crop)
            if torch.cuda.is_available():
                model.cuda()
                ms_up = ms_up.cuda()
                ms = ms.cuda()
                pan = pan.cuda()
            # out = netG(z)  # 生成图片
            # out = model(pan_crop, ms_gray_crop)
            start = time.time()
            out= model(ms_up, ms, pan)
            end = time.time()
        output = out.cpu()

        time_ave = (end - start) / batch_size
        print('Average testing time is', time_ave)

        for i in range(batch_size):
            # image = (output.data[i] + 1) / 2.0
            count += 1
            image = output.data[i]
            # image = image.mul(255).byte()
            image = np.transpose(image.numpy(), (1, 2, 0))

            if sate == 'wv3_8':
                save_f_name = sate + '_%03d.mat' % (count)
            else:
                save_f_name = sate + '_%03d.tif' % (count)
            save_image_RS(sate, save_f_name, image) ### used for save the images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--satellite', type=str, default='pl')  # satellite here
    parser.add_argument('--checkpoint', type=str, default='ik_model_epoch_500.pth')
    parser.add_argument('--dataset_dir', type=str, default='E:\\remote sense image fusion\\Source Images\\')
    parser.add_argument('--ratio', type=int, default=4)  # ratio here
    parser.add_argument("--net", default='FusionNet')
    parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--cuda', default=False, help='use cuda?')
    opt = parser.parse_args()

    model = Generator(1).cuda().eval()   ########## need to change

    # opt.satellite = 'wv3_8'
    opt.satellite = 'pl'

    if opt.satellite == 'ik':
        opt.checkpoint = 'netG_ik_epoch_1_1300.pth'
    elif opt.satellite == 'pl':
        opt.checkpoint = 'netG_pl_epoch_1_1300.pth'
    elif opt.satellite == 'wv3_8':
        opt.checkpoint = 'netG_wv3_8_epoch_1_1300.pth'

    dataset_dir = 'E:\\remote sense image fusion\\Source Images\\'
    model_path = r'.\model1\\'

    test_set = TestDatasetFromFolder(dataset_dir, opt.satellite, upscale_factor=1)  # 测试集导入
    # test_set = get_test_set(opt.dataset_dir, opt.satellite, opt.ratio)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                  shuffle=False)
    checkpoint = torch.load(model_path + opt.satellite + '/%s' % opt.checkpoint,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    # model.eval()
    # r'E:\remote sense image fusion\compared_methods\2021-FusionNet\FusionNet-main'
    #                r'\FSnet_ik_pl_lu\models\%s' % opt.checkpoint,
    #                map_location=lambda storage, loc: storage)

    # image_path = r'../fused'
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)

    test(test_data_loader, model, opt.satellite)
