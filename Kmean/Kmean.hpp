#ifndef KMEAN_HPP
#define KMEAN_HPP


#include<opencv2/core.hpp>
#include<cmath>
#include<iostream>


/*@brief n维数据处理，数据矩阵中的行向量为数据的属性.
* @param data 数据矩阵，N×n
* @param x 聚类个数
* @param label l×N的标签矩阵
* @param center 聚类中心，x×n，每一行都一个聚类中心
*/
int ndimKmean(const cv::Mat& data, int x, cv::Mat& label, cv::Mat& center);

/*@brief 主函数
* @param image 输入图像,CV_8UC3
* @param x 聚类个数
* @param dst 输入图像，彩色，CV_8UC3
* @param label 输出的标签矩阵,CV_8UC1
*/
int kmeans(const cv::Mat& image, int x, cv::Mat& dst, cv::Mat& label);




/*@brief 欧式距离函数
* @param pixel 输入像素的头地址
* @param mid 输入聚类中心
*/
double dist(uchar* pixel, double* mid);


/*@brief 图像着色
* @param label 标签矩阵
* @param midpixel 聚类中心颜色器
* @param dst 输出聚类图像
*/
int drawColor(cv::Mat& label, cv::Mat& midpixel,cv::Mat& dst);




class Kmean
{
	friend int kmeans(const cv::Mat& image, int x, cv::Mat& dst, cv::Mat& label);
public:





    /*@brief 初始化聚类对象,将聚类中心随机化,其midpixel为n×3的矩阵
    * @param x 聚类个数
    */
	Kmean(int n);



	/*@brief 传入输入图像，初始化标签矩阵
	* @param image 输入图像 
	*/
	int init(cv::Mat& image);




	/*@brief 更新标签
	*/
	int update_pixel();
	
	
	
	
	/*@brief 更新聚类中心
	*/
	int update_center();
private:
	//源图像
	cv::Mat imagec;
	//标记图
	cv::Mat label;
	//中心聚类像素n×3
	cv::Mat midpixel; 
};

#endif // !KMEAN_HPP
