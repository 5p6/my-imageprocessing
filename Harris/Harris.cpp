#include"Harris.hpp"
int DeletePoints(cv::Mat& image,std::vector<cv::Point>& contours,int distance,double T)
{
	//确保distance是一个奇数
	if (distance % 2 == 0)
	{
		distance++;
	}
	contours.clear();
	(contours).swap(contours);
	double max = 0;
	cv::minMaxLoc(image, NULL, &max, NULL, NULL);
	std::vector<double>myarray(distance * distance,0);
	std::vector<double>myarrayback(distance * distance, 0);
	double* ptr = NULL;
	double* pixel = NULL;
	int mrow = image.rows;
	int mcol = image.cols;
	int size = (distance - 1) / 2;
	int index = 0;

	for (int i = 1; i < mrow - 1; i++)
	{
		pixel = image.ptr<double>(i);
		for (int j = 1; j < mcol - 1; j++)
		{
			
			//抽取(i,j)点在领域[-size,size]的点
			for (int k = -size; k <= size ; k++)
			{
				if (i + k <= 0 || i + k >= mrow) continue;
				ptr = image.ptr<double>(i + k);
				for (int f = -size; f <= size; f++)
				{
					if (j + f <= 0 || j + f >= mcol) continue;
					myarray[index] = *(ptr + j + f);
					index++;
				}
			}
			sort(myarray.begin(), myarray.end());
			

			//如果自己是最大值
			if (*(pixel + j) == myarray[distance*distance-1]) {
				//并且大于一个梯度阈值
				if (*(pixel + j) > T * max)
				{
					contours.push_back(cv::Point(j, i));
				}
			}
			index = 0;
			myarray.swap(myarrayback);
		}
	}
	return 1;
}

void HarrisquickSort(double* r, int begin, int end)
{
	int i = begin, j = end;
	double tem;
	if (begin < end)
	{
		tem = r[begin];
		while (i != j)
		{
			while (j > i && r[j] >= tem) j--;
			r[i] = r[j];
			while (j > i && r[i] <= tem)i++;
			r[j] = r[i];
		}
		r[i] = tem;
		HarrisquickSort(r, begin, i - 1);
		HarrisquickSort(r, i + 1, end);
	}
}

int Harris::Sobel(cv::Mat& image,cv::Mat& imagex, cv::Mat& imagey )
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "输入错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	imagex.release();
	imagey.release();
	imagex.create(imagec.size(), CV_64FC1);
	imagex.setTo(0);
	imagey.create(imagec.size(), CV_64FC1);
	imagey.setTo(0);



	uchar* pixelup = NULL;
	uchar* pixeldown = NULL;
	uchar* pixel = NULL;

	double* pixelx = NULL;
	double* pixely = NULL;

	for (int i = 1; i < imagec.rows - 1; i++)
	{
		pixelup = imagec.ptr<uchar>(i - 1);
		pixel = imagec.ptr<uchar>(i);
		pixeldown = imagec.ptr<uchar>(i + 1);

		pixelx = imagex.ptr<double>(i);
		pixely = imagey.ptr<double>(i);
		for (int j = 1; j < imagec.cols - 1; j++)
		{
			*(pixelx + j) =
				-(*(pixelup + j - 1)) - (*(pixelup + j)) * 2 - (*(pixelup + j + 1)) +
				(*(pixeldown + j - 1)) + (*(pixeldown + j)) * 2 + (*(pixeldown + j + 1));

			*(pixely + j) =
				-(*(pixelup + j - 1)) - (*(pixel + j - 1)) * 2 - (*(pixeldown + j - 1)) +
				(*(pixelup + j + 1)) + (*(pixel + j + 1)) * 2 + (*(pixeldown + j + 1));
		}
	}
	return 1;
}

int GaussianKenrel(cv::Mat& gauusian, double var, cv::Size size)
{

	//生成的高斯核是以方差为一个像素单位的,即每个像素间的dx或者dy距离是一个方差var，而非固定的1值.
	if (size.height % 2 == 0 || size.width % 2 == 0)
	{
		std::cout << "高斯核大小错误" << std::endl;
		return -1;
	}
	gauusian.release();
	gauusian.create(size, CV_64FC1);
	double* pixel = NULL;
	int mrows = size.height;
	int mcols = size.width;
	cv::Point center = cv::Point((mrows - 1) / 2, (mcols - 1) / 2);
	for (int i = 0; i < mrows; i++)
	{
		pixel = gauusian.ptr<double>(i);
		for (int j = 0; j < mcols; j++)
		{
			*(pixel + j) = (exp(-(((i - center.x) * (i - center.x) + (j - center.y) * (j - center.y)) / (2 * var * var)))) / 2 * CV_PI / (var * var);
		}
	}

	//归一化
	double sum = cv::sum(gauusian)[0];
	gauusian = gauusian / sum;
	return 1;
}

template<class _Tp>
int Harris::Kernel(const cv::Mat& image,cv::Mat& dst,cv::Mat& kernel)
{

	//_Tp = image.type();
	//输入函数的类型就是模板的类型
	if (image.empty() || image.channels() != 1)
	{
		return -1;
	}
	if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0)
	{
		std::cout << "滤波器核大小错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	//重置输出矩阵
	dst.release();
	dst = imagec.clone();
	dst.setTo(0);


	int mrows = kernel.rows;
	int mcols = kernel.cols;
	//填充
	cv::Mat img = cv::Mat(cv::Size((mcols - 1) + imagec.cols, (mrows - 1) + imagec.rows), imagec.type(), cv::Scalar(0));
	_Tp* imgptr = NULL;
	_Tp* srcptr = NULL;
	//填充
	for (int i = (int)(mrows - 1) / 2; i < imagec.rows + (mrows - 1) / 2; i++)
	{
		imgptr = img.ptr<_Tp>(i);
		srcptr = imagec.ptr<_Tp>(i - (mrows - 1) / 2);
		for (int j = (int)(mcols - 1) / 2; j < image.cols + (mcols - 1) / 2; j++)
		{
			*(imgptr + j) = *(srcptr + j - (mcols - 1) / 2);
		}
	}


	_Tp* dstptr = NULL;//输入图像的指针
	double* kptr = NULL;
	//检测
	double msum = 0;

	for (int i = (int)(mrows - 1) / 2; i < imagec.rows + (mrows - 1) / 2; i++)
	{

		dstptr = dst.ptr<_Tp>(i - (mrows - 1) / 2);
		//检测行的指针
		for (int j = (int)(mcols - 1) / 2; j < image.cols + (mcols - 1) / 2; j++)
		{
			// 0~mi行  0~mj列
			for (int mi = 0; mi < kernel.rows; mi++)
			{
				imgptr = img.ptr<_Tp>(i - (mrows - 1) / 2 + mi);
				kptr = kernel.ptr<double>(mi);
				for (int mj = 0; mj < kernel.cols; mj++)
				{
					msum += (*(kptr + mj)) * (*(imgptr + j - (mcols - 1) / 2 + mj));
				}
			}
			*(dstptr + j - (mcols - 1) / 2) = cv::saturate_cast<_Tp>(msum);
			msum = 0;
		}
	}
	return 1;
}

int conv3(const cv::Mat& image, cv::Mat& dst, cv::Mat& w)
{
	if (image.empty()||image.channels()!=1)
	{
		return -1;
	}
	cv::Mat imagec = image.clone();
	double* ptrup = NULL;
	double* ptr = NULL;
	double* ptrdown = NULL;

	
	dst.release();
	dst.create(imagec.size(), CV_64FC1);
	double* dstptr = NULL;

	if (w.type() != CV_64FC1) {
		w.convertTo(w, CV_64F);
	}
	double* wptrup = w.ptr<double>(0);
	double* wptr = w.ptr<double>(1);
	double* wptrdown = w.ptr<double>(2);
	int mrows = imagec.rows;
	int mcols = imagec.cols;

	for (int i = 1; i < mrows - 1; i++)
	{
		ptrup = imagec.ptr<double>(i - 1);
		ptr = imagec.ptr<double>(i);
		ptrdown = imagec.ptr <double> (i + 1);

		dstptr = dst.ptr<double>(i);
		for (int j = 1; j < mcols - 1; j++)
		{

			*(dstptr + j) =cv::saturate_cast<double>(
				*(ptrup + j - 1)* (* wptrup)+*(ptrup + j) * ( * (wptrup + 1)) + *(ptrup + j + 1) * ( * (wptrup + 2)) +
				*(ptr + j - 1) * (* wptr) + *(ptr + j) * ( * (wptr + 1) )+ *(ptr + j + 1) * ( * (wptr + 2)) +
				*(ptrdown + j - 1) * ( * wptrdown)+*(ptrdown + j) * ( * (wptrdown + 1)) + *(ptrdown + j + 1) * ( * (wptrdown + 2))
				);
		}
	}
	return 1;
}


int MaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,double T)
{

	double max = 0;
	cv::minMaxLoc(image, NULL, &max, NULL, NULL);
	double* pixelup = NULL;
	double* pixel = NULL;
	double* pixeldown = NULL;
	std::vector<double> myarray;
	myarray.resize(9);
	for (int i = 1; i < image.rows - 1; i++)
	{
		double* pixelup = image.ptr<double>(i - 1);
		double* pixel = image.ptr<double>(i);
		double* pixeldown = image.ptr<double>(i + 1);
		for (int j = 1; j < image.cols - 1; j++)
		{
			//排序
			myarray[0] = *(pixelup + j - 1); myarray[3] = *(pixelup + j); myarray[6] = *(pixelup + j + 1);
			myarray[1] = *(pixel + j - 1); myarray[4] = *(pixel + j); myarray[7] = *(pixel + j + 1);
			myarray[2] = *(pixeldown + j - 1); myarray[5] = *(pixeldown + j); myarray[8] = *(pixeldown + j + 1);
			sort(myarray.begin(), myarray.end());

			//如果自己是最大值
			if (*(pixel + j) == myarray[8]) {
				//并且大于一个梯度阈值
				if (*(pixel + j) > T * max)
				{
					contours.push_back(cv::Point(j, i));
				}
			}
		}
	}
	return 1;
}

int meMaxLoc(cv::Mat& image,std::vector<cv::Point>& contours,int max,double T)
{

	double maxlevel = 0;
	cv::minMaxLoc(image, NULL, &maxlevel, NULL, NULL);
	double* pixelupup = NULL;
	double* pixelup = NULL;
	double* pixel = NULL;
	double* pixeldown = NULL;
	double* pixeldowndown = NULL;

	double myarray[25] = {};

	int mrow = image.rows;
	int mcol = image.cols;
	int index = 0;


	for (int i = 2; i < mrow - 2; i++)
	{
		pixelupup = image.ptr<double>(i - 2);
		pixelup = image.ptr<double>(i - 1);
		pixel = image.ptr<double>(i);
		pixeldown = image.ptr<double>(i + 1);
		pixeldowndown = image.ptr<double>(i + 2);
		for (int j = 2; j < mcol - 2; j++)
		{
			//排序
			myarray[0] = *(pixelupup + j - 2);     myarray[5] = *(pixelupup + j - 1);        myarray[10] = *(pixelupup + j );       myarray[15] = *(pixelupup + j + 1);       myarray[20] = *(pixelupup + j + 2);
			myarray[1] = *(pixelup + j - 2);       myarray[6] = *(pixelup + j - 1);          myarray[11] = *(pixelup + j );	        myarray[16] = *(pixelup + j + 1);		  myarray[21] = *(pixelup + j + 2);
			myarray[2] = *(pixel + j - 2);         myarray[7] = *(pixel + j - 1);            myarray[12] = *(pixeldown + j );       myarray[17] = *(pixel + j + 1);	          myarray[22] = *(pixel + j + 2);
			myarray[3] = *(pixeldown + j - 2);     myarray[8] = *(pixeldown + j - 1);        myarray[13] = *(pixel + j );	        myarray[18] = *(pixeldown + j + 1);		  myarray[23] = *(pixeldown + j + 2);
			myarray[4] = *(pixeldowndown + j - 2); myarray[9] = *(pixeldowndown + j - 1);    myarray[14] = *(pixeldowndown + j );   myarray[19] = *(pixeldowndown + j + 1);	  myarray[24] = *(pixeldowndown + j + 2);
			
			HarrisquickSort(myarray, 0, 24);

			//如果自己是最大值并且大于一个梯度阈值
			if (*(pixel + j) == myarray[24] && *(pixel + j) > T * maxlevel) {
				contours[index]=cv::Point(j, i);
				index++;
				if (index == max )
				{
					break;
				}
			}
		}
		if (index == max )
		{
			break;
		}
	}
	
	return index;
}

int memaxLoc(cv::Mat& image, std::vector<cv::Point>& contours,int max ,double T)
{

	double maxlevel = 0;
	cv::minMaxLoc(image, NULL, &maxlevel, NULL, NULL);
	double* pixelup = NULL;
	double* pixel = NULL;
	double* pixeldown = NULL;
	double myarray[9] = {};

	int mrow = image.rows;
	int mcol = image.cols;
	int index = 0;
	for (int i = 1; i < mrow - 1; i++)
	{
		double* pixelup = image.ptr<double>(i - 1);
		double* pixel = image.ptr<double>(i);
		double* pixeldown = image.ptr<double>(i + 1);
		for (int j = 1; j < mcol - 1; j++)
		{
			//排序
			myarray[0] = *(pixelup + j - 1); myarray[3] = *(pixelup + j); myarray[6] = *(pixelup + j + 1);
			myarray[1] = *(pixel + j - 1); myarray[4] = *(pixel + j); myarray[7] = *(pixel + j + 1);
			myarray[2] = *(pixeldown + j - 1); myarray[5] = *(pixeldown + j); myarray[8] = *(pixeldown + j + 1);
			HarrisquickSort(myarray, 0, 8);


			//如果自己是最大值并且大于一个梯度阈值
			if (*(pixel + j) == myarray[8] && *(pixel + j) > T * maxlevel) {
				contours[index] = cv::Point(j, i);
				index++;
				if (index == max)
				{
					break;
				}
			}
		}
		if (index == max)
		{
			break;
		}
	}
	return index;
}
Harris::Harris(cv::Mat& image)
{
	cv::Mat fx, fy;
	//Sobel梯度函数
	Sobel(image, fx, fy);
	//平方图计算
	fxx = fx.mul(fx);
	fxy = fx.mul(fy);
	fyy = fy.mul(fy);
	//加权高斯核的滤波
	cv::Mat w;
	GaussianKenrel(w, 1, cv::Size(3, 3));
	conv3(fxx, fxx, w);
	conv3(fxy, fxy, w);
	conv3(fyy, fyy, w);
}

Harris::Harris()
{}

int HarrisCorner(const cv::Mat& src, std::vector<cv::Point>& contours, int max ,double k , double T)
{
	//image必须是传入的灰度图,CV_8UC1
	//contours是装载角点的坐标
	//k是敏感因子 区间为[0.15,0.01]
	//T是角点阈值 区间为[0.4,0.01]
	cv::Mat imagec;
	
	if (src.empty())
	{
		system("pause");
		return -1;
	}

	if (src.channels() == 3)
	{
		cv::cvtColor(src, imagec, cv::COLOR_BGR2GRAY);
	}
	else
	{
		imagec = src.clone();
	}

	//传入imagec图像,然后将梯度图像得到，并且得到权重函数.
	Harris* harr = new Harris(imagec);
	//点集重新指定大小
	contours.resize(max);
	//三个辅佐指针
	double* fxxptr = NULL;
	double* fyyptr = NULL;
	double* fxyptr = NULL;

	//响应值矩阵
	cv::Mat resimg = cv::Mat(imagec.size(), CV_64FC1);
	double* rptr = NULL;

	int mrows = imagec.rows;
	int mcols = imagec.cols;
	for (int i = 1; i < mrows - 1; i++)
	{
		fxyptr = harr->fxy.ptr<double>(i);
		fxxptr = harr->fxx.ptr<double>(i);
		fyyptr = harr->fyy.ptr<double>(i);
		rptr = resimg.ptr<double>(i);
		for (int j = 1; j < mcols - 1; j++)
		{
			double m = (*(fxxptr + j)) * (*(fyyptr + j)) - (*(fxyptr + j)) * (*(fxyptr + j));
			double trace = fxxptr[j] + fyyptr[j];
			*(rptr + j) = m - k * trace * trace;
		}
	}
	//DeletePoints(resimg, contours , 5 , T);
	int index=meMaxLoc(resimg, contours, max , T);
	if (index < max)
	{
		contours.resize(index);
	}
	delete harr;
	return 1;
}
