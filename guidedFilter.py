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
    imgDst = np.zeros([img_h, img_w], np.float32)

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
    float_I = np.float32(I)
    mean_II = boxFilter(float_I**2, r) / N
    ##mean_II = boxFilter(I*I, r) / N   ##这里I如果是uint8的,I*I的结果会溢出255,所以转为float32处理
    var_I = mean_II - mean_I**2
    
    mean_p = boxFilter(p, r) / N
    float_Ip = float_I * np.float32(p)
    mean_Ip = boxFilter(float_Ip, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    ##这里del是尽快释放一些空间，图像不大的话可以不用
    del mean_Ip, mean_II

    a = cov_Ip / (var_I + eps)
    del cov_Ip, var_I
    b = mean_p - a * mean_I
    del mean_p, mean_I

    mean_a = boxFilter(a, r) / N
    mean_b = boxFilter(b, r) / N

    outputImg = mean_a * I + mean_b
    return outputImg

 
