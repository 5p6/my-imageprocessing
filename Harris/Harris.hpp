#ifndef HARRIS_HPP
#define HARRIS_HPP


#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<vector>
#include<stack>


/*@brief ɾ���Ͻ��Ľǵ�
@param contours �ǵ㼯��
@param distance ��С����
*/
int DeletePoints(cv::Mat& image,std::vector<cv::Point>& contours,double distance,double T);



/*@brief 3��3���
@param image ����ͼ��
@param dst ���ͼ��
@param w �������� 3��3
*/
int conv3(const cv::Mat& image, cv::Mat& dst, cv::Mat& w);



/*@brief ��������
@param r ����ͷ��ַ
@param begin ��ʼλ��
@param end ����λ��
*/
void HarrisquickSort(double* r, int begin, int end);


/*@brief ��˹����ȡ
@param gauusian ���������˹��
@param var ����
@param size ��˹�˴�С
*/
int GaussianKenrel(cv::Mat& gauusian, double var, cv::Size size);




/*@brief �Ǽ���ֵ����
@param image ����ͼ��
@param contours ����㼯
@param T ��ֵ
*/
int MaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,double T);


/*@briefС�Ķ��ķǼ���ֵ����
@param image ����ͼ��
@param contours ����㼯
@param T ��ֵ
*/
int meMaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,int max,double T);
class Harris
{
public:
	/*@brief sobel�ݶȺ���
	@param image ����ͼ��
	@param imagex dx�����ͼ��
	@param imagey dy�����ͼ��
	*/
	int static Sobel(
		cv::Mat& image,//����
		cv::Mat& imagex, //dx��� CV_64F
		cv::Mat& imagey //dy��� CV_64F
	);


	/*@brief ���Ծ������(ֻҪ�������ߴ�ĺ˶�����ʹ��)
	@param image ����ͼ��
	@param dst ���ͼ��
	@param kernel �����
	*/
	template<class _Tp>
	int Kernel(const cv::Mat& image,cv::Mat& dst,cv::Mat& kernel);


public:
	Harris();//Ĭ�Ϲ���
	Harris(cv::Mat& image);//�вι���
	cv::Mat fxx;//ͼ��dx^2�ݶ�
	cv::Mat fyy;//ͼ��dy^2�ݶ�
	cv::Mat fxy;//ͼ��dx*dy�ݶ�
};

/*@brief ��harris�ǵ�
@param image CV_8UC3 or CV_8UC1 else can not;
@param contours output points vector.
@param k ��������
@param T ��ֵ����
*/
int HarrisCorner(const cv::Mat& image, std::vector<cv::Point>& contours, int max = 500,double k = 0.1, double T = 0.01);
#endif