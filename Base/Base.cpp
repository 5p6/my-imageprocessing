#include"Base.hpp"


int Me::calchist(const cv::Mat& image, std::vector<double>& a, int k)
{
	if (image.empty())
	{
		std::cout << "无图像" << std::endl;
		return -1;
	}
	a.clear();
	(a).swap(a);
	a.resize(256);
	int mrows = image.rows;
	int mcols = image.cols;
	cv::Mat imagec = image.clone();
	for (int i = 0; i < mrows; i++)
	{
		uchar* p = imagec.ptr<uchar>(i);
		for (int j = 0; j < mcols; j++)
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
int Me::thresthold(const cv::Mat& image) //图像阈值;
{
	if (image.empty() || image.channels() != 1)
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



	double m = 0;//全局均值
	for (int i = 0; i < 256; i++)
	{
		m += i * hist[i];
	}


	std::vector<double> var(256, 0);//类间方差

	for (int i = 0; i < 256; i++)//类间方差计算
	{
		if (0 < P1[i] && P1[i] < 1)
		{
			var[i] = ((m * P1[i] - m1[i]) * (m * P1[i] - m1[i])) / (P1[i] * (1 - P1[i]));
		}
	}


	int k = 0;
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
	if (image.empty() || image.channels() != 1)
	{
		return -1;
	}

	cv::Mat imagec = image.clone();
	dst.release();
	int mrows = imagec.rows;
	int mcols = imagec.cols;


	std::vector<uchar> mid(9,0);
	const uchar* pixelup = NULL;
	const uchar* pixel = NULL;
	const uchar* pixeldown = NULL;

	//中间商
	cv::Mat src = imagec.clone();
	uchar* srcptr = NULL;



	for (int i = 1; i < mrows - 1; i++)
	{

		pixelup = imagec.ptr<uchar>(i - 1);
		pixel = imagec.ptr<uchar>(i);
		pixeldown = imagec.ptr<uchar>(i + 1);

		srcptr = src.ptr<uchar>(i);

		for (int j = 1; j < mcols - 1; j++)
		{



			mid[0]=(*(pixelup + j - 1));       mid[1]=(*(pixelup + j));       mid[2]=(*(pixelup + j + 1));
			mid[3]=(*(pixel + j - 1));         mid[4]=(*(pixel + j));         mid[5]=(*(pixel + j + 1));
			mid[6]=(*(pixeldown + j - 1));     mid[7]=(*(pixeldown + j));     mid[8]=(*(pixeldown + j + 1));


			sort(mid.begin(), mid.end());
			*(srcptr + j) = mid[4];
		}
	}


	dst = src.clone();
	src.release();
	return 1;

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
	cv::Mat P1(cv::Size(256, 256), CV_64FC1, cv::Scalar(0));
	cv::Mat P2 = P1.clone(), P3 = P1.clone();

	float* p1 = NULL;
	float* p2 = NULL;
	float* p3 = NULL;
	//类的均值矩阵 和  其指针
	cv::Mat m1(cv::Size(256, 256), CV_64FC1, cv::Scalar(0));
	cv::Mat m2 = m1.clone(), m3 = m1.clone();

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
				*(p2 + k2) += hist[i];
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
					*(n1 + k2) += (float)i * hist[i] / (*(p1 + k2));
				}
			}

			if (*(p2 + k2) != 0) {
				for (int i = k1 + 1; i <= k2; i++)
				{
					*(n2 + k2) += (float)i * hist[i] / (*(p2 + k2));
				}
			}

			if (*(p3 + k2) != 0) {
				for (int i = k2 + 1; i < 256; i++)
				{
					*(n3 + k2) += (float)i * hist[i] / (*(p3 + k2));
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
int Me::gobalold(const cv::Mat& image, int dt)
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "图片输入错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	std::vector<double> hist;
	Me::calchist(imagec, hist, calc::Not);
	double T = 0;
	for (int i = 0; i < hist.size(); i++)
	{
		T += i * hist[i];
	}
	T = T / ((float)imagec.rows * imagec.cols);

	double m1 = 0, m2 = 0; //两个类的均值
	double sum1 = 0, sum2 = 0;//两个类的像素个数

	double T1 = 0;//记录前一个阈值

	while (true)
	{
		T1 = T;
		for (int i = 0; i < (int)T; i++)
		{
			sum1 += hist[i];
			m1 += hist[i] * i;
		}
		m1 = m1 / sum1;//求第一个类的均值

		for (int i = (int)T; i < hist.size(); i++)
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
int Me::Sobel(const cv::Mat& image, cv::Mat& dst, cv::Mat& sobel)
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
	int mrows = image.rows;
	int mcols = image.cols;
	for (int i = 1; i < mrows - 1; i++)
	{
		pixelup = imagec.ptr(i - 1);
		pixel = imagec.ptr(i);
		pixeldown = imagec.ptr(i + 1);
		dstptr = dst.ptr(i);
		for (int j = 1; j < mcols - 1; j++)
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
int Me::GaussianKenrel(cv::Mat& gauusian, double var, cv::Size size)
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
	cv::Point center = cv::Point((size.height - 1) / 2, (size.width - 1) / 2);
	for (int i = 0; i < size.height; i++)
	{
		pixel = gauusian.ptr<double>(i);
		for (int j = 0; j < size.width; j++)
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
int Me::Kernel(const cv::Mat& image, cv::Mat& dst, cv::Mat& kernel)
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
int Me::CovMatrix(cv::Mat& mat, cv::Mat& cov, cv::Mat& mean)
{

	//mat的每一行都是一个向量，为行向量.
	if (mat.empty())
	{
		return 0;
	}
	cv::Mat m;
	mat.convertTo(m, CV_32F);
	int mrows = m.rows;
	int mcols = m.cols;



	mean.release();
	mean = cv::Mat::zeros(cv::Size(mcols, 1), CV_32FC1);
	float* meanptr = NULL;


	cov.release();
	cov = cv::Mat::zeros(cv::Size(mcols, mcols), CV_32FC1);

	//每行向量
	cv::Mat p(cv::Size(mcols, 1), CV_32FC1);
	cv::Mat diffrow, diffcol;



	float* ptr = NULL;
	//均值
	for (int i = 0; i < mcols; i++)
	{
		//meanptr = mean.ptr<float>(0);
		p = m.rowRange(i, i + 1);
		mean += p;
		//for (int j = 0; j < mrows; j++)
		//{
		//	ptr = m.ptr<float>(j);
		//	(*(meanptr + i)) += *(ptr + i) / mrows;
		//}
	}
	mean = mean / mrows;


	//开始求解协方差矩阵
	for (int i = 0; i < mrows; i++)
	{
		p = m.rowRange(i, i + 1);
		diffrow = p - mean;
		cv::transpose(diffrow, diffcol);
		cov += (diffcol * diffrow) / (mrows - 1);
	}
	return 1;
}
int Me::white_balance(const cv::Mat& image, cv::Mat& dst)
{
	if(image.channels()!=3)
	{
		return -1;
	}
	cv::Mat imagec = image.clone();
	dst.release();
	int mrows = imagec.rows;
	int mcols = imagec.cols;
	float sumnum = mrows * mcols;


	uchar* pixel = NULL;
	double mean_b = 0;
	double mean_g = 0;
	double mean_r = 0;

	for (int i = 0; i < mrows; i++)
	{
		pixel = imagec.ptr<uchar>(i);
		for (int j = 0; j< mcols;j++)
		{
			mean_b += *(pixel + 3 * j);
			mean_g += *(pixel + 3 * j + 1);
			mean_r += *(pixel + 3 * j + 2);
		}
	}

	mean_b = mean_b / sumnum;
	mean_g = mean_g / sumnum;
	mean_r = mean_r / sumnum;

	double avr = (mean_b + mean_g + mean_r) / 3;

	double b = avr / mean_b;
	double g = avr / mean_g;
	double r = avr / mean_r;

	dst.create(imagec.size(), CV_8UC3);
	uchar* dstptr = NULL;

	for (int i = 0; i < mrows; i++)
	{
		dstptr = dst.ptr<uchar>(i);
		pixel = imagec.ptr<uchar>(i);
		for (int j = 0; j < mcols; j++)
		{
			*(dstptr + 3 * j) = (*(pixel + 3 * j)) * b;
			*(dstptr + 3 * j + 1) = (*(pixel + 3 * j + 1)) * g;
			*(dstptr + 3 * j + 2) = (*(pixel + 3 * j + 2)) * r;
		}
	}


	return 1;

}
int Me::Gamma(const cv::Mat& image, cv::Mat* dst, double gamma)
{
	if (image.empty())
	{
		return -1;
	}
	



}
int filelist(std::string& filename, std::vector<std::string>& filelist)
{
	if (filename.empty())
	{
		std::cout << "文件名消失,请重新输入" << std::endl;
		return -1;
	}
	




}
int threshinto(const cv::Mat& image, cv::Mat& dst, uchar x)//图像的otsu阈值处理
{
	//_Tp为输入图像的元素数据类型
	if (image.empty() || image.channels() != 1)
	{
		std::cout << "无图像输入" << std::endl;
		system("pause");
		return -1;
	}
	cv::Mat imagec = image.clone();

	dst.release();
	dst = imagec.clone();
	dst.setTo(0);
	int mrows = imagec.rows;
	int mcols = imagec.cols;


	uchar* pixel = NULL;
	uchar* dstp = NULL;

	for (int i = 0; i < mrows; i++)
	{
		pixel = imagec.ptr<uchar>(i);
		dstp = dst.ptr<uchar>(i);
		for (int j = 0; j < mcols; j++)
		{
			if (pixel[j] > x)
			{
				*(dstp + j)= 255;
			}
			else
			{
				*(dstp + j )= 0;
			}
		}
	}
	return 1;
}
int doubleinto(const cv::Mat& image, cv::Mat& dst, std::vector<int>& num, int k )
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
		uchar p[3] = { 0,125,255 };
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
		sobel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
		Me::Sobel(image, dst, sobel);
		break;
	}
	default:
		break;
	}

	return 1;

}
int Sobel(cv::Mat& image, cv::Mat& imagex, cv::Mat& imagey)
{
	if (image.empty() || image.type() != CV_8UC1)
	{
		std::cout << "输入错误" << std::endl;
		return -1;
	}
	cv::Mat imagec = image.clone();
	imagex.release();
	imagey.release();
	imagex.create(imagec.size(), CV_8UC1);
	imagex.setTo(0);
	imagey.create(imagec.size(), CV_8UC1);
	imagey.setTo(0);



	uchar* pixelup = NULL;
	uchar* pixeldown = NULL;
	uchar* pixel = NULL;

	uchar* pixelx = NULL;
	uchar* pixely = NULL;

	for (int i = 1; i < imagec.rows - 1; i++)
	{
		pixelup = imagec.ptr<uchar>(i - 1);
		pixel = imagec.ptr<uchar>(i);
		pixeldown = imagec.ptr<uchar>(i + 1);

		pixelx = imagex.ptr<uchar>(i);
		pixely = imagey.ptr<uchar>(i);
		for (int j = 1; j < imagec.cols - 1; j++)
		{
			*(pixelx + j) = cv::saturate_cast<uchar>(
				-(*(pixelup + j - 1)) - (*(pixelup + j)) << 1 - (*(pixelup + j + 1)) +
				(*(pixeldown + j - 1)) + (*(pixeldown + j)) << 1 + (*(pixeldown + j + 1)));

			*(pixely + j) = cv::saturate_cast<uchar>(
				-(*(pixelup + j - 1)) - (*(pixel + j - 1)) << 1 - (*(pixeldown + j - 1)) +
				(*(pixelup + j + 1)) + (*(pixel + j + 1)) << 1 + (*(pixeldown + j + 1)));
		}
	}
	return 1;
}
