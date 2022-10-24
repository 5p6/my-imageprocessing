#pragma once
#include<opencv2/core/core.hpp>
#include<iostream>
#include<vector>
#include<math.h>
#include<set>
#include<algorithm>
namespace calc {
	int Yes = 1;
	int Not = 0;

}
namespace Thresh {
	int Color = 3;
	int Gray = 1;
}
namespace sobel
{
	int dx = 0;
	int dy = 1;
}
class Me
{
public:
	static int thresthold(const cv::Mat& image);//otsu阈值
	static int gobalold(const cv::Mat& image,int dt);//全局阈值
	static int calchist(const cv::Mat& image, std::vector<double>& a,int k =calc::Yes);//直方图，容器也可以用map或者set.
	static int midfilter(const cv::Mat& image, cv::Mat& dst);//中值滤波
	static int Sobel(const cv::Mat& image, cv::Mat& dst, cv::Mat& sobel);//sobel梯度计算
	static int doubleold(const cv::Mat& image, std::vector<int>& num);//otsu双阈值
};


class pooling
{
public:
	static int meanpool(const cv::Mat& image,cv::Mat& dst);
	static int maxpool(const cv::Mat& image, cv::Mat& dst,int size);


};
int Me::calchist(const cv::Mat& image, std::vector<double>& a,int k )
{
	if (image.empty())
	{
		std::cout << "无图像" << std::endl;
		return -1;
	}
	a.clear();
	a.resize(256);
	cv::Mat imagec = image.clone();
	for (int i = 0; i < imagec.rows; i++)
	{
		uchar* p = imagec.ptr<uchar>(i);
		for (int j = 0; j < imagec.cols; j++)
		{
			if (p[j] >= 0 && p[j] <= 255)
			{
				a[p[j]]++;
			}
		}
	}

	switch (k)
	{
	    case 1:
	   {
		   double area = (double)imagec.rows * imagec.cols;
		   for (auto it = a.begin(); it != a.end(); it++)
		   {
			    (*it) = (*it) / area;
		   };
		   break;
	    }
	    case 0:
		    break;
	    default:
	     {
		    std::cout << "输入错误" << std::endl;
		    break;
	      }
	}
	return 1;
}
int Me::thresthold(const cv::Mat& image)//图像阈值;
{
	if (image.empty() || image.type() !=CV_8UC1)
	{
		std::cout << "图像输入错误" << std::endl;
		system("pause");
		return -1;
	}

	cv::Mat imagec = image.clone();
	std::vector<double> hist;
	Me::calchist(imagec, hist);//得到归一化直方图容器

    std::vector<double> P1(256, 0);//类c1的发生概率
    std::vector<double> m1(256, 0);//类c1的均值
    for (int i = 0; i < 256; i++)
    {
    	for (int k = 0; k < i + 1; k++)
    	{
    		P1[i] += hist[k];
    	}
    }
     for (int i = 0; i < 256; i++)
     {
     	for (int k = 0; k < i + 1; k++)
     	{
     			m1[i] += k * hist[k];
     	}
	 }

     
     
     double m=0;//全局均值
     for (int i = 0; i < 256; i++)
     {
     	m += i * hist[i];
     }
     
     
     std::vector<double> var(256, 0);//类间方差
     
     for (int i = 0; i < 256; i++)//类间方差计算
     {
     	if (0<P1[i]  &&   P1[i]<1)
     	{
     		var[i] =( (m * P1[i] - m1[i]) * (m * P1[i] - m1[i]) ) / (P1[i] * (1 - P1[i]));
     	}
    }
     
     
       int k=0;
     //寻找让类间方差最大的k值
     for (int i = 0; i < 256; i++)
     {
     	if (var[i] > var[k])
     	{
     		k = i;
     	}
     }
     
     return k;
     
}
int Me::midfilter(const cv::Mat& image, cv::Mat& dst)
{
	if(image.empty()||image.channels()!=1)	
	{ 
		return -1;
	}

	cv::Mat imagec=image.clone();
	dst.release();



	std::vector<uchar> mid;
	const uchar* pixelup = NULL;
	const uchar* pixel = NULL;
	const uchar* pixeldown = NULL;

	//中间商
	cv::Mat src=image.clone();
	uchar* srcptr = NULL;



	for (int i = 1; i < imagec.rows - 1; i++)
	{

		 pixelup = imagec.ptr<uchar>(i-1);
		 pixel = imagec.ptr<uchar>(i );
		 pixeldown = imagec.ptr<uchar>(i + 1);

		 srcptr = src.ptr<uchar>(i);

		 for (int j = 1; j < imagec.cols - 1; j++)
		 {



			 mid.push_back(*(pixelup + j - 1));       mid.push_back(*(pixelup + j));     mid.push_back(*(pixelup + j + 1));
			 mid.push_back(*(pixel + j - 1));           mid.push_back(*(pixel + j));           mid.push_back(*(pixel + j + 1));
			 mid.push_back(*(pixeldown + j - 1));  mid.push_back(*(pixeldown + j)); mid.push_back(*(pixeldown + j + 1));
			 
			 
			 sort(mid.begin(), mid.end());
			 *(srcptr + j) = mid[4];
			  mid.clear();
		}
	}


	dst = src;
	return 1;

}
int pooling::meanpool(const cv::Mat& image, cv::Mat& dst)
{
	if (image.empty()||image.channels()!=1)
	{
		return -1;
	}
	cv::Mat imagec = image.clone();
	uchar* pixelup = NULL;
	uchar* pixel = NULL;

	uchar* dstptr = NULL;


	dst.release();

	int height = imagec.rows / 2;
	int width = imagec.cols / 2;

	dst.create(cv::Size(width, height), CV_8UC1);

	for (int i = 1; i < imagec.rows - 2; i+=2)
	{
		pixelup = imagec.ptr<uchar>(i - 1);
		pixel = imagec.ptr<uchar>(i);

		dstptr = dst.ptr<uchar>((i-1)/2);

		for (int j = 1; j < imagec.cols - 2; j+=2)
		{
			*(dstptr + (j-1)/2) = cv::saturate_cast<uchar>
				(
					( *(pixel + j - 1) + *(pixel + j) + *(pixelup + j) + *(pixelup + j - 1) )/ 4
				);

		}

	}


	return 1;


}
int pooling::maxpool(const cv::Mat& image, cv::Mat& dst,int size)
{
	if (image.empty() || image.channels() != 1 || size%2==0)
	{
		return -1;
	}
	std::vector<uchar> m;
	//sort(m.begin(), m.end());


	cv::Mat imagec = image.clone();
	dst.release();
	dst.create(cv::Size((int)(image.cols / size),(int) (image.rows / size)), CV_8UC1);
	const uchar* pixel = NULL;
	uchar* dstptr = NULL;
	for (int i = 1; i < image.rows - 1; i+=size)
	{
		pixel = imagec.ptr<uchar>(i);
		dstptr = imagec.ptr<uchar>(i+(i-1)/size);

		for (int j = 1; j < image.cols; j += size)

		{







		}




	}


}
int Me::doubleold(const cv::Mat& image, std::vector<int>& num)//otsu双阈值法
{
	if (image.channels() != 1 || image.empty())
	{
		return -1;
	}

	cv::Mat imagec = image.clone();
	//归一化直方图
	std::vector<double>hist;
	Me::calchist(imagec, hist);

	//全局均值
	double m = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		m += i * hist[i];
	}
	//双阈值变量
	int var1 = 0;
	int var2 = 0;
	
	//k1为行，k2为列;
	//类的概率矩阵 和 其指针
	cv::Mat P1(cv::Size(256, 256), CV_64FC1,cv::Scalar(0));
	cv::Mat P2 = P1.clone(), P3 = P1.clone();

	float* p1 = NULL;
	float* p2 = NULL;
	float* p3 = NULL;
	//类的均值矩阵 和  其指针
	cv::Mat m1(cv::Size(256, 256), CV_64FC1,cv::Scalar(0));
    cv::Mat m2 = m1.clone(),m3 = m1.clone();

	float* n1 = NULL;
	float* n2 = NULL;
	float* n3 = NULL;
	//类间方差矩阵
	cv::Mat var(cv::Size(256, 256), CV_64FC1, cv::Scalar(0));
	
	float* varptr = NULL;


	//三个类的概率
	for (int k1 = 1; k1 < 254; k1++)  
	{
		p1 = P1.ptr<float>(k1);
		p2 = P2.ptr<float>(k1);
		p3 = P3.ptr<float>(k1);
		for (int k2 = k1 + 1; k2 < 255; k2++)
		{
			for (int i = 0; i <= k1; i++)
			{
				*(p1 + k2) += hist[i];
			}

			for (int i = k1 + 1; i <= k2; i++)
			{
				*(p2+ k2) += hist[i];
			}
			*(p3 + k2) = 1 - *(p1 + k2) - *(p2 + k2);
		}
	}


	//三个类的均值矩阵
	for (int k1 = 1; k1 < 254; k1++)
	{
		n1 = m1.ptr<float>(k1);
		n2 = m2.ptr<float>(k1);
		n3 = m3.ptr<float>(k1);
		p1 = P1.ptr<float>(k1);
		p2 = P2.ptr<float>(k1);
		p3 = P3.ptr<float>(k1);

		for (int k2 = k1 + 1; k2 < 255; k2++)
		{

			if (*(p1 + k2) != 0) {
				for (int i = 0; i <= k1; i++)
				{
					*(n1 + k2) += (float) i * hist[i] / (*(p1 + k2));
				}
			}

			if (*(p2 + k2) != 0) {
				for (int i = k1 + 1; i <= k2; i++)
				{
					*(n2 + k2) += (float) i * hist[i] / (*(p2 + k2));
				}
			}

			if (*(p3 + k2) != 0) {
				for (int i = k2 + 1; i < 256; i++)
				{
					*(n3 + k2) +=(float) i * hist[i] / (*(p3 + k2));
				}
			}
		}
	}


	for (int k1 = 1; k1 < 254; k1++)
	{
		n1 = m1.ptr<float>(k1);
		n2 = m2.ptr<float>(k1);
		n3 = m3.ptr<float>(k1);
		p1 = P1.ptr<float>(k1);
		p2 = P2.ptr<float>(k1);
		p3 = P3.ptr<float>(k1);

		varptr = var.ptr<float>(k1);
		for (int k2 = k1 + 1; k2 < 255; k2++)
		{

			*(varptr + k2) =
				(*(p1 + k2)) * (*(n1 + k2) - m) * (*(n1 + k2) - m) +
				(*(p2 + k2)) * (*(n2 + k2) - m) * (*(n2 + k2) - m) +
				(*(p3 + k2)) * (*(n3 + k2) - m) * (*(n3 + k2) - m);
		}
	}


	float* me = var.ptr<float>(var1) + var2;//初始化最大值点

	for (int k1 = 1; k1 < 254; k1++)
	{
		varptr = var.ptr<float>(k1);
		for (int k2 = k1 + 1; k2 < 255; k2++)
		{
			if (*(varptr + k2) > *me)
			{
				//更新最大值的坐标
				var1 = k1;
				var2 = k2;
				me = var.ptr<float>(var1) + var2;//更新
			}
		}
	}

	num.push_back(var1);
	num.push_back(var2);
	
	return 1;

}
int Me::gobalold(const cv::Mat& image,int dt)
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "图片输入错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	std::vector<double> hist;
	Me::calchist(imagec, hist,calc::Not);
	double T = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		T += i * hist[i];
	}
	T = T / ((float)imagec.rows * imagec.cols);

	double m1=0, m2=0; //两个类的均值
	double sum1=0, sum2 = 0;//两个类的像素个数

	double T1=0;//记录前一个阈值
	
	while (true)
	{
		T1 = T;
	    for (int i = 0; i < (int)T; i++)
	    {
	    	sum1 += hist[i];
	    	m1 += hist[i] * i ;
	    }
	    m1 = m1 / sum1;//求第一个类的均值

	    for (int i = (int)T ; i <hist.size(); i++)
	    {
	    	sum2 += hist[i];
	    	m2 += hist[i] * i;
	    }
	    m2 = m2 / sum2;//第二个类的均值

	    T = (m1 + m2) / 2;//重新计算阈值

		if (abs(T - T1) <= dt)break;
		//清零
		m1 = 0;
		m2 = 0;
		sum1 = 0;
		sum2 = 0;

	}
	return (int)T;
}
int Me::Sobel(const cv::Mat& image, cv::Mat& dst,cv::Mat& sobel)
{
	cv::Mat imagec = image.clone();
	dst.release();
	dst = cv::Mat::zeros(imagec.size(), CV_8UC1);
	float* sobup = sobel.ptr<float>(0);
	float* sob = sobel.ptr<float>(1);
	float* sobdown = sobel.ptr<float>(2);
	
	uchar* pixelup = NULL;
	uchar* pixel = NULL;
	uchar* pixeldown = NULL;
	uchar* dstptr = NULL;

	float num = 0;
	for (int i = 1; i < image.rows - 1; i++)
	{
		pixelup = imagec.ptr(i - 1);
		pixel = imagec.ptr(i);
		pixeldown = imagec.ptr(i + 1);
		dstptr = dst.ptr(i);
		for (int j = 1; j < image.cols - 1; j++)
		{
			for (int k = -1; k < 2; k++)
			{
				num += *(pixelup + j + k) * (*(sobup + k + 1)) + *(pixel + j + k) * (*(sob + k + 1)) + *(pixeldown + j + k) * (*(sobdown + k + 1));
			}
			*(dstptr + j) = cv::saturate_cast<uchar>(num);
			num = 0;
		}
	}

	return 1;





}
int threshinto(const cv::Mat& image, cv::Mat& dst, uchar x)//图像的otsu阈值处理
{
	if (image.empty() || image.type()!=CV_8UC1)
	{
		std::cout << "无图像输入" << std::endl;
		system("pause");
		return -1;
	}
	cv::Mat imagec = image.clone();
	dst.release();
	dst = imagec.clone();
	dst.setTo(0);


	for (int i = 0; i < imagec.rows; i++)
	{
		uchar* pixel = imagec.ptr<uchar>(i);
		uchar* dstp = dst.ptr<uchar>(i);
		for (int j = 0; j < imagec.cols; j++)
		{
			if (pixel[j] > x)
			{
				dstp[j] = 255;
			}
			else
			{
				dstp[j] = 0;
			}
		}
	}
	return 1;
}
int doubleinto(const cv::Mat& image, cv::Mat& dst, std::vector<int>& num,int k = Thresh::Color)
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "输入错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	dst.release();
	

	
	cv::RNG* P = new cv::RNG;
	switch (k)
	{
	case 3:
	{
		dst.create(imagec.size(), CV_8UC3);
		std::vector<cv::Vec3b> p;
		for (int i = 0; i < 3; i++)
		{
			cv::Vec3b pixel;
			pixel[0] = P->uniform(i * 3, 255 - i);
			pixel[1] = P->uniform(i * 3, 255 - i);
			pixel[2] = P->uniform(i * 3, 255 - i);
			p.push_back(pixel);
		}
		delete P;


		cv::Vec3b* pixel = NULL;
		uchar* imgptr = NULL;
		for (int i = 0; i < image.rows; i++)
		{
			imgptr = imagec.ptr<uchar>(i);
			pixel = dst.ptr<cv::Vec3b>(i);
			for (int j = 0; j < image.cols; j++)
			{
				if (*(imgptr + j) <= num[0])
				{
					*(pixel + j) = p[0];
				}
				else if (*(imgptr + j) > num[0] && *(imgptr + j) <= num[1])
				{
					*(pixel + j) = p[1];
				}
				else
				{
					*(pixel + j) = p[2];
				}
			}
		}
		break;
	}
	case 1:
	{
		dst.create(imagec.size(), CV_8UC1);
		uchar p[3]={0,125,255};
		uchar* dstptr = NULL;
		uchar* imgptr = NULL;

		for (int i = 0; i < imagec.rows; i++)
		{
			imgptr = imagec.ptr<uchar>(i);
		    dstptr = dst.ptr<uchar>(i);
			for (int j = 0; j < imagec.cols; j++)
			{
				if (*(imgptr + j) <= num[0])
				{
					*(dstptr + j) = p[0];
				}
				else if (*(imgptr + j) > num[0] && *(imgptr + j) <= num[1])
				{
					*(dstptr + j) = p[1];
				}
				else
				{
					*(dstptr + j) = p[2];
				}
			}
		}
	}
	default:
		break;
	}


	return 1;
}
int My_sobel(const cv::Mat& image, cv::Mat& dst, int x)
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "输入图片错误" << std::endl;
		return -1;
	}
	cv::Mat sobel;
	switch (x)
	{
	case 0:
	{
        sobel = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
		Me::Sobel(image, dst, sobel);
		break;
	}
	case 1:
	{
		sobel = (cv::Mat_<float>(3, 3) << -1,0,1,-2,0,2,-1,0,1);
		Me::Sobel(image, dst, sobel);
		break;
	}
	default:
		break;
	}

	return 1;

}
