#include"guidedFilter.h"

int32 main(int32 argc, char* argv[])
{
	string leftImgPath = "D:\\temp1\\left1.png";
	string rightImgPath = "D:\\temp1\\right1.png";

	Mat leftImg = imread(leftImgPath, IMREAD_GRAYSCALE);
	Mat rightImg = imread(rightImgPath, IMREAD_GRAYSCALE);

	//namedWindow("srcImg", WINDOW_NORMAL);
	//imshow("srcImg", leftImg);
	//namedWindow("cum", WINDOW_NORMAL);
	//imshow("cum", rightImg);
	//waitKey(0);

	Mat leftImgFiltered = guidedFilter(leftImg, leftImg, 3, 0.01);
	Mat rightImgFiltered = guidedFilter(rightImg, rightImg, 5, 0.01);

	string leftImgOutPath = "D:\\temp1\\left1_GF.png";
	string rightImgOutPath = "D:\\temp1\\right1_GF.png";
	vector<int>saveParams;
	saveParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	saveParams.push_back(0);

	imwrite(leftImgOutPath, leftImgFiltered, saveParams);
	imwrite(rightImgOutPath, rightImgFiltered, saveParams);


	printf("After Guided Filter...\n");
	getchar();
	return 0;
}