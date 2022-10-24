#ifndef HARRIS_HPP
#define HARRIS_HPP


#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<vector>
#include<stack>


/*@brief 删除较近的角点
@param contours 角点集合
@param distance 最小距离
*/
int DeletePoints(cv::Mat& image,std::vector<cv::Point>& contours,double distance,double T);



/*@brief 3×3卷积
@param image 输入图像
@param dst 输出图像
@param w 输入卷积核 3×3
*/
int conv3(const cv::Mat& image, cv::Mat& dst, cv::Mat& w);



/*@brief 快速排序法
@param r 数组头地址
@param begin 开始位置
@param end 结束位置
*/
void HarrisquickSort(double* r, int begin, int end);


/*@brief 高斯核求取
@param gauusian 输入输出高斯核
@param var 方差
@param size 高斯核大小
*/
int GaussianKenrel(cv::Mat& gauusian, double var, cv::Size size);




/*@brief 非极大值抑制
@param image 输入图像
@param contours 输入点集
@param T 阈值
*/
int MaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,double T);


/*@brief小改动的非极大值抑制
@param image 输入图像
@param contours 输入点集
@param T 阈值
*/
int meMaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,int max,double T);
class Harris
{
public:
	/*@brief sobel梯度函数
	@param image 输入图像
	@param imagex dx的输出图像
	@param imagey dy的输出图像
	*/
	int static Sobel(
		cv::Mat& image,//输入
		cv::Mat& imagex, //dx输出 CV_64F
		cv::Mat& imagey //dy输出 CV_64F
	);


	/*@brief 线性卷积函数(只要是奇数尺寸的核都可以使用)
	@param image 输入图像
	@param dst 输出图像
	@param kernel 卷积核
	*/
	template<class _Tp>
	int Kernel(const cv::Mat& image,cv::Mat& dst,cv::Mat& kernel);


public:
	Harris();//默认构造
	Harris(cv::Mat& image);//有参构造
	cv::Mat fxx;//图像dx^2梯度
	cv::Mat fyy;//图像dy^2梯度
	cv::Mat fxy;//图像dx*dy梯度
};

/*@brief 求harris角点
@param image CV_8UC3 or CV_8UC1 else can not;
@param contours output points vector.
@param k 比例因子
@param T 阈值因子
*/
int HarrisCorner(const cv::Mat& image, std::vector<cv::Point>& contours, int max = 500,double k = 0.1, double T = 0.01);
#endif