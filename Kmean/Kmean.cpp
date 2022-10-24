#include"Kmean.hpp"


double dist(float* pixel, double* mid)
{
	double d1 = (*pixel - *mid) * (*pixel - *mid);
	double d2 = (*(pixel + 1) - *(mid + 1)) * (*(pixel + 1) - *(mid + 1));
	double d3 = (*(pixel + 2) - *(mid + 2)) * (*(pixel + 2) - *(mid + 2));
	double distance = sqrt(d1 + d2 + d3);
	return distance;
}



int drawColor(cv::Mat& label, cv::Mat& midpixel ,cv::Mat& dst)
{
	dst.create(label.size(), CV_8UC3);
	int mrows = label.rows;
	int mcols = label.cols;

	uchar* dstptr = NULL;
	double* labptr = NULL;
	double* midptr = NULL;


	for (int i = 0; i < mrows; i++)
	{
		labptr = label.ptr<double>(i);
		dstptr = dst.ptr<uchar>(i);
		for (int j = 0; j < mcols; j++)
		{
			midptr = midpixel.ptr<double>(*(labptr + j));
			*(dstptr + 3 * j + 0) = (uchar) * (midptr);
			*(dstptr + 3 * j + 1) = (uchar) * (midptr + 1);
			*(dstptr + 3 * j + 2) = (uchar) * (midptr + 2);
		}
	}
	return 1;
}


Kmean::Kmean(int n)
{
	cv::RNG rng;
	this->midpixel.create(cv::Size(3, n), CV_64FC1);
	rng.fill(midpixel, cv::RNG::UNIFORM, 0, 255);
}



int Kmean::init(cv::Mat& image)
{
	this->imagec = image.clone();
	imagec.convertTo(imagec, CV_32F);
	label.release();
	label.create(imagec.size(), CV_64FC1);
	label.setTo(-1);
	return 1;
}


int Kmean::update_pixel()
{
	
	int mrows = imagec.rows;
	int mcols = imagec.cols;
	int n = midpixel.rows;

	int index = -1;

	double distance = 0;
	double mindist = 9999999;

	double* labptr = NULL;
	float* pixel = NULL;
	double* midptr = NULL;

	for (int i = 0; i < mrows; i++)
	{
		labptr = label.ptr<double>(i);
		pixel = imagec.ptr<float>(i);
		for (int j = 0; j < mcols; j++)
		{
			for (int t = 0; t < n; t++)
			{
				midptr = midpixel.ptr<double>(t);
				distance = dist((pixel + 3 * j), midptr);
				if (mindist > distance)
				{
					mindist = distance;
					index = t;
				}
			}
			*(labptr + j) = index;
			index = -1;
			mindist = 9999999;
		}
	}
	return 1;
}



int Kmean::update_center()
{
	int mrows = imagec.rows;
	int mcols = imagec.cols;
	int n = midpixel.rows;
	double* midptr = NULL;
	std::vector<int> count(n, 0);

	double* labptr = NULL;
	float* pixel = NULL;
	for (int i = 0; i < mrows; i++)
	{
		labptr = label.ptr<double>(i);
		pixel = imagec.ptr<float>(i);
		for (int j = 0; j < mcols; j++)
		{
			count[(int) * (labptr + j)]++;
			midptr = midpixel.ptr<double>(*(labptr + j));
			*(midptr) += *(pixel + 3*j );
			*(midptr + 1) += *(pixel + 3 * j + 1);
			*(midptr + 2) += *(pixel + 3 * j + 2);
		}
	}
	for (int i = 0; i < n; i++)
	{
		if (count[i] != 0) {
			midptr = midpixel.ptr<double>(i);
			*(midptr) /= count[i];
			*(midptr + 1) /= count[i];
			*(midptr + 2) /= count[i];
		}
	}
	return 1;
}




int kmeans(const cv::Mat& image,int x,cv::Mat& dst,cv::Mat& label)
{
	if (image.empty() || image.channels()!=3)
	{
		system("pause");
		return -1;
	}
	Kmean* k = new Kmean(x);
	cv::Mat imagec = image.clone();
	
	dst.release();
	label.release();

	k->init(imagec);

	//迭代20次
	for (int i = 0; i < 8; i++)
	{
		k->update_pixel();
		k->update_center();
	}
	
	label = k->label.clone();
	drawColor(label, k->midpixel, dst);
	delete k;
	return 1;
}








int ndimKmean(const cv::Mat& data, int x, cv::Mat& label, cv::Mat& center)
{
	if (data.empty())
	{
		return -1;
		std::cout << "无数据" << std::endl;
	}















}
