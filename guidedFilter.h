#ifndef GUIDEDFILTER_H_
#define GUIDEDFILTER_H_

#include<iostream>
#include<vector>
#include<math.h>
#include <numeric>

#include <omp.h>
#include <string.h>
using namespace std;

#include <opencv2/opencv.hpp> 
using namespace cv;


////设定数据类型
typedef unsigned char uint8;
typedef unsigned char int8;
typedef unsigned short uint16;
typedef short int16;
typedef int int32;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;
typedef long long int64;
typedef unsigned long long uint64;


Mat boxFilter(Mat &srcImg, int32 r);
Mat guidedFilter(Mat &I, Mat &p, int32 r, float32 eps);

#endif