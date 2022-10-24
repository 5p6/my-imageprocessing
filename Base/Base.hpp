#ifndef BASE_H
#define BASE_H

//头文件包括
#include<opencv2/core/core.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<math.h>
#include<algorithm>
#include<fstream>

//主要函数
class Me
{
public:
	enum calc
	{
		Yes = 1,
		Not = 0
	};
	enum Thresh {
		Color = 3,
		Gray = 1
	};
	enum sobel
	{
		dx = 0,
		dy = 1
	};
public:
	static int thresthold(const cv::Mat& image);//otsu阈值
	static int gobalold(const cv::Mat& image, int dt);//全局阈值
	static int calchist(const cv::Mat& image, std::vector<double>& a, int k = calc::Yes);//直方图，容器也可以用map或者set.
	static int midfilter(const cv::Mat& image, cv::Mat& dst);//中值滤波
	static int Sobel(const cv::Mat& image, cv::Mat& dst, cv::Mat& sobel);//sobel梯度计算
	static int doubleold(const cv::Mat& image, std::vector<int>& num);//otsu双阈值
	static int GaussianKenrel(cv::Mat& gauusian, double var, cv::Size size);
	static int CovMatrix(cv::Mat& mat, cv::Mat& cov, cv::Mat& mean);
	template<class _Tp>
	static int Kernel(const cv::Mat& image, cv::Mat& dst, cv::Mat& kernel);//任意奇数大小滤波器核的卷积
	static int white_balance(const cv::Mat& image, cv::Mat& dst);
	static int Gamma(const cv::Mat& image, cv::Mat* dst, double gamma);
	static int filelist(std::string& filename, std::vector<std::string>& filelist);
};

int threshinto(const cv::Mat& image, cv::Mat& dst, uchar x);
int doubleinto(const cv::Mat& image, cv::Mat& dst, std::vector<int>& num, int k = Me::Thresh::Color);
int My_sobel(const cv::Mat& image, cv::Mat& dst, int x);
int Sobel(cv::Mat& image, cv::Mat& imagex, cv::Mat& imagey);
#endif