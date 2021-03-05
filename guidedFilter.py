#-*-coding: utf-8-*-
##author: wxs  Wuhan University
##data: 2021-3-5
##Implementation of Guided Filter by KaiMing He

import numpy as np
import cv2
from matplotlib import pyplot as plt

##利用box filter来快速地计算每个窗口内的像素总和
##Box Filter通过积分图的思想将这一过程优化，从O(MN)变成O(4)
##输入：引导图像和窗口的半径
def boxFilter(imgSrc, r):
    img_h, img_w = imgSrc.shape
    imgDst = np.zeros([img_h, img_w])

    ##首先累加y方向(↓)
    imgCum = np.cumsum(imgSrc, 0)
    ##注意:通过不同位置的累加值的差相减来得到每个像素在当前窗口内，y方向上的窗口累加值
    imgDst[0:r + 1, :] = imgCum[r:2*r + 1, :]
    imgDst[r + 1:img_h-r, :] = imgCum[2*r + 1:img_h, :] - imgCum[0:img_h-2*r - 1, :]
    imgDst[img_h - r:img_h, :] = np.tile(imgCum[img_h-1, :].reshape([1, -1]), [r, 1]) - imgCum[img_h-2*r - 1:img_h-r - 1, :]

    ##然后在累加完y方向的基础上再按x方向(→)累加，来实现整个窗口二维的累加
    imgCum = np.cumsum(imgDst, 1)
    ##同样通过x方向不同位置的累加值的差来计算窗口内所有像素的和
    imgDst[:, 0:r+1] = imgCum[:, r:2*r+1]
    imgDst[:, r+1:img_w-r] = imgCum[:, 2*r+1:img_w] - imgCum[:, 0:img_w-2*r-1]
    imgDst[:, img_w-r:] = np.tile(imgCum[:, img_w-1].reshape([-1, 1]), [1, r]) - imgCum[:, img_w-2*r-1:img_w-r-1]

    return imgDst

##输入：待滤波图像、引导图像、输出图像、窗口半径、
def guidedFilter(I, p, r, eps):
    img_h, img_w = p.shape
    ##N矩阵的每个像素计算了当前像素作为窗口中心像素时窗口内像素的个数
    N = boxFilter(np.ones_like(I), r)

    mean_I = boxFilter(I, r) / N
    mean_II = boxFilter(I ** 2, r) / N
    var_I = mean_II - mean_I**2
    
    mean_p = boxFilter(p, r) / N
    mean_Ip = boxFilter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    ##这里del是尽快释放一些空间，图像不大的话可以不用
    del mean_Ip

    a = cov_Ip / (var_I + eps)
    del cov_Ip, var_I
    b = mean_p - a * mean_I
    del mean_p, mean_I

    mean_a = boxFilter(a, r) / N
    mean_b = boxFilter(b, r) / N

    outputImg = mean_a * I + mean_b
    return outputImg


####cones image
##leftImgPath = r'D:\work_space\Codes\SGM\cones\left.png'
##rightImgPath = r'D:\work_space\Codes\SGM\cones\right.png'
##
##leftImg = cv2.imread(leftImgPath, cv2.IMREAD_GRAYSCALE)
##rightImg = cv2.imread(rightImgPath, cv2.IMREAD_GRAYSCALE)
##
####r = 2
####eps = 0.1**2
##
##res = []
##rs = [2, 4, 8]
##epss = [0.1**2, 0.2**2, 0.4**2]
##for r in rs:
##    for eps in epss:
##        filtered = guidedFilter(leftImg, leftImg, r, eps)
##        res.append(filtered)
##
##plt.subplot(331), plt.imshow(res[0], 'gray'), plt.subplot(332), plt.imshow(res[1], 'gray'), plt.subplot(333), plt.imshow(res[2], 'gray')
##plt.subplot(334), plt.imshow(res[3], 'gray'), plt.subplot(335), plt.imshow(res[4], 'gray'), plt.subplot(336), plt.imshow(res[5], 'gray')
##plt.subplot(337), plt.imshow(res[6], 'gray'), plt.subplot(338), plt.imshow(res[7], 'gray'), plt.subplot(339), plt.imshow(res[8], 'gray')
##plt.show()
  
