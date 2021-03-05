#include "guidedFilter.h"

//将矩阵按照某一个方向逐个累加，axis==0 沿着y轴累加(↓)，axis==1沿着x轴累加(→)
template<class T>
Mat accumulateSum(Mat &srcImg, int32 axis = 0)
{
	int32 img_h = srcImg.rows, img_w = srcImg.cols;
	Mat cumImg = Mat::zeros(img_h, img_w, CV_32FC1);

	//从上而下累加
	if (axis == 0)
	{
		//第0行
	//#pragma omp parallel for
	//	for (int32 j = 0; j < img_w; ++j)
	//	{
	//		cumImg.at<float32>(0, j) = srcImg.at<T>(0, j);
	//	}
		srcImg.row(0).copyTo(cumImg.row(0));

		//第1行到最后一行
		for (int32 i = 1; i < img_h; ++i)
		{
	#pragma omp parallel for
			for (int32 j = 0; j < img_w; ++j)
			{
				cumImg.at<float32>(i, j) = cumImg.at<float32>(i - 1, j) + srcImg.at<T>(i, j);
			}
		}
	}
	////从左到右累加
	else
	{
	//#pragma omp parallel for
	//	for (int32 i = 0; i < img_h; ++i)
	//	{
	//		cumImg.at<float32>(i, 0) = srcImg.at<T>(i, 0);
	//	}
		srcImg.col(0).copyTo(cumImg.col(0));

		for (int32 j = 1; j < img_w; ++j)
		{
	#pragma omp parallel for
			for (int32 i = 0; i < img_h; ++i)
			{
				cumImg.at<float32>(i, j) = cumImg.at<float32>(i, j - 1) + srcImg.at<T>(i, j);
			}
		}
	}
	return cumImg;
}

Mat matAccumulateSum(Mat &srcImg, int32 axis = 0)
{
	int32 dType = srcImg.depth();
	if (dType == CV_8S)
	{
		return accumulateSum<int8>(srcImg, axis);
	}
	else if (dType == CV_16U)
	{
		return accumulateSum<uint16>(srcImg, axis);
	}
	else if (dType == CV_16S)
	{
		return accumulateSum<int16>(srcImg, axis);
	}
	else if (dType == CV_32S)
	{
		return accumulateSum<int32>(srcImg, axis);
	}
	else if (dType == CV_32F)
	{
		return accumulateSum<float32>(srcImg, axis);
	}
	else
	{
		return accumulateSum<uint8>(srcImg, axis);
	}
}

Mat boxFilter(Mat &srcImg, int32 r)
{
	int32 img_h = srcImg.rows, img_w = srcImg.cols;
	Mat dstImg = Mat::zeros(img_h, img_w, CV_32F);
	Mat cumImg = matAccumulateSum(srcImg, 0);

	//按y方向累加值计算窗口元素y方向上的和
	cumImg.rowRange(r, 2 * r + 1).copyTo(dstImg.rowRange(0, r + 1));
	Mat res = cumImg.rowRange(2 * r + 1, img_h) - cumImg.rowRange(0, img_h - 2 * r - 1);
	res.copyTo(dstImg.rowRange(r + 1, img_h - r));
	for (int32 k1 = img_h - r, k2 = img_h - 2 * r - 1; k1 < img_h && k2 < img_h - r - 1; ++k1, ++k2)
	{
		Mat tmp = cumImg.rowRange(img_h - 1, img_h) - cumImg.rowRange(k2, k2 + 1);
		tmp.copyTo(dstImg.rowRange(k1, k1 + 1));
	}

	//按x方向累加值计算窗口元素x方向上的和
	cumImg = matAccumulateSum(dstImg, 1);
	cumImg.colRange(r, 2 * r + 1).copyTo(dstImg.colRange(0, r + 1));
	res = cumImg.colRange(2 * r + 1, img_w) - cumImg.colRange(0, img_w - 2 * r - 1);
	res.copyTo(dstImg.colRange(r + 1, img_w - r));
	for (int32 k1 = img_w - r, k2 = img_w - 2 * r - 1; k1 < img_w && k2 < img_w - r - 1; ++k1, ++k2)
	{
		Mat tmp = cumImg.colRange(img_w - 1, img_w) - cumImg.colRange(k2, k2 + 1);
		tmp.copyTo(dstImg.colRange(k1, k1 + 1));
	}

	return dstImg;
}

Mat guidedFilter(Mat &I, Mat &p, int32 r, float32 eps)
{
	int32 img_h = I.rows, img_w = I.cols;
	Mat oneMat = Mat::ones(img_h, img_w, CV_8UC1);
	Mat N = boxFilter(oneMat, r);
	
	Mat mean_I = boxFilter(I, r) / N;
	Mat float_I;
	I.convertTo(float_I, CV_32FC1);
	Mat II = float_I.mul(float_I);
	Mat mean_II = boxFilter(II, r) / N;
	Mat var_I = mean_II - mean_I.mul(mean_I);
	II.release();
	mean_II.release();
	
	Mat mean_p = boxFilter(p, r) / N;
	Mat float_p;
	p.convertTo(float_p, CV_32FC1);
	Mat Ip = float_I.mul(float_p);
	float_p.release();
	Mat mean_Ip = boxFilter(Ip, r) / N;
	Ip.release();
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	mean_Ip.release();

	Mat a = cov_Ip / (var_I + eps);
	Mat b = mean_p - a.mul(mean_I);
	cov_Ip.release();
	var_I.release();
	mean_p.release();
	mean_I.release();

	Mat mean_a = boxFilter(a, r) / N;
	Mat mean_b = boxFilter(b, r) / N;

	a.release();
	b.release();

	Mat outputImg = mean_a.mul(float_I) + mean_b;
	return outputImg;
}

