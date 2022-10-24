#ifndef KMEAN_HPP
#define KMEAN_HPP


#include<opencv2/core.hpp>
#include<cmath>
#include<iostream>


/*@brief nά���ݴ������ݾ����е�������Ϊ���ݵ�����.
* @param data ���ݾ���N��n
* @param x �������
* @param label l��N�ı�ǩ����
* @param center �������ģ�x��n��ÿһ�ж�һ����������
*/
int ndimKmean(const cv::Mat& data, int x, cv::Mat& label, cv::Mat& center);

/*@brief ������
* @param image ����ͼ��,CV_8UC3
* @param x �������
* @param dst ����ͼ�񣬲�ɫ��CV_8UC3
* @param label ����ı�ǩ����,CV_8UC1
*/
int kmeans(const cv::Mat& image, int x, cv::Mat& dst, cv::Mat& label);




/*@brief ŷʽ���뺯��
* @param pixel �������ص�ͷ��ַ
* @param mid �����������
*/
double dist(uchar* pixel, double* mid);


/*@brief ͼ����ɫ
* @param label ��ǩ����
* @param midpixel ����������ɫ��
* @param dst �������ͼ��
*/
int drawColor(cv::Mat& label, cv::Mat& midpixel,cv::Mat& dst);




class Kmean
{
	friend int kmeans(const cv::Mat& image, int x, cv::Mat& dst, cv::Mat& label);
public:





    /*@brief ��ʼ���������,���������������,��midpixelΪn��3�ľ���
    * @param x �������
    */
	Kmean(int n);



	/*@brief ��������ͼ�񣬳�ʼ����ǩ����
	* @param image ����ͼ�� 
	*/
	int init(cv::Mat& image);




	/*@brief ���±�ǩ
	*/
	int update_pixel();
	
	
	
	
	/*@brief ���¾�������
	*/
	int update_center();
private:
	//Դͼ��
	cv::Mat imagec;
	//���ͼ
	cv::Mat label;
	//���ľ�������n��3
	cv::Mat midpixel; 
};

#endif // !KMEAN_HPP
