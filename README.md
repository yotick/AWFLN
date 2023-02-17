# AWFLN: An Adaptive Weighted Feature Learning Network for Pansharpening-TGRS2023

For training please run train.py, and for testing please run the file test_RS.py for reduced scale images and test_lu_full.py for full scale image. The file test_lu_full_crop.py is used when the images are too big. 

We would be pleased if you can cite this paper, and please refer to:

    @ARTICLE{10034991,
    author={Lu, Hangyuan and Yang, Yong and Huang, Shuying and Chen, Xiaolong and Chi, Biwei and Liu, Aizhu and Tu, Wei},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={AWFLN: An Adaptive Weighted Feature Learning Network for Pansharpening}, 
    year={2023},
    volume={61},
    number={},
    pages={1-15},
    abstract={Deep learning (DL)-based pansharpening methods have shown great advantages in extracting spectral–spatial features from multispectral (MS) and panchromatic (PAN) images compared with traditional methods. However, most DL-based methods ignore the local inner connection between the source images and the high-resolution MS (HRMS) image, which cannot fully extract spectral–spatial information and attempt to improve the quality of fusion by increasing the complexity of the network. To solve these problems, a lightweight network based on adaptive weighted feature learning network (AWFLN) is proposed for pansharpening. Specifically, a novel detail extraction model is first built by exploring the local relationship between HRMS and source images, thereby improving the accuracy of details and the interpretability of the network. Guided by this model, we then design a residual multiple receptive-field structure to fully extract spectral–spatial features of source images. In this structure, an adaptive feature learning block based on spectral–spatial interleaving attention is proposed to adaptively learn the weights of features and improve the accuracy of the extracted details. Finally, the pansharpened result is obtained by a detail injection model in AWFLN. Numerous experiments are carried out to validate the effectiveness of the proposed method. Compared to traditional and state-of-the-art methods, AWFLN performs the best both subjectively and objectively, with high efficiency. The code is available at https://github.com/yotick/AWFLN.},
    keywords={},
    doi={10.1109/TGRS.2023.3241643},
    ISSN={1558-0644},
    month={},}

