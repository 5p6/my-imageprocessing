#include"cutout.hpp"

//显示图像
cv::Mat showpicure;
//复制图像
cv::Mat picure;
//鼠标的边界
cv::Mat contour;
//漫水填充的图像
cv::Mat roi;
//窗口名称
std::string Windows_name;
//回调函数标志
int CallBack_flags = 1;


/*
鼠标回调函数

*/
void press(int event, int y, int x, int flags, void* param)
{
	//像素
	cv::Vec3b* pixel = NULL;
	cv::Vec3b* picptr = NULL;
	//当鼠标是按下的时候
	if (flags == cv::EVENT_FLAG_LBUTTON)
	{
		for (int i = -2; i < 2; i++) {
			if (y + i < 0)continue;
			for (int j = -2; j < 2; j++) {
				if (x + j < 0)continue;
				pixel = contour.ptr<cv::Vec3b>(x + i);
				picptr = showpicure.ptr<cv::Vec3b>(x + i);

				*(pixel + y + j) = cv::Vec3b(0, 0, 255);
				*(picptr + y + j) = cv::Vec3b(0, 0, 255);
			}
		}
		cv::imshow(Windows_name, showpicure);
	}
	if (event == cv::EVENT_LBUTTONUP)
	{
		CallBack_flags = 0;
	}
		
}


int find_contour(cv::Mat& dst,int value )
{
	//坎尼检测子图像
	cv::Mat canny;
	cv::Canny(picure, canny, value, 2 * value, 3, false);
	////轮廓点集
	std::vector<std::vector<cv::Point>> contours;
	////找外轮廓
	cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	//
	////由canny检测子产生边界的图像
	cv::Mat draw = cv::Mat::zeros(picure.size(), CV_8UC3);
	cv::drawContours(draw, contours, -1, cv::Scalar(0, 0, 255));

	//漫水图
	roi.release();
	roi.create(picure.size(), CV_8UC1);
	uchar* roiptr = NULL;
	int mrows = roi.rows;
	int mcols = roi.cols;
	
	cv::Vec3b* conptr = NULL;
	cv::Vec3b* drawptr = NULL;
	
	//得到边界
	for (int i = 0; i < mrows; i++) {
		conptr = contour.ptr<cv::Vec3b>(i);
		drawptr = draw.ptr<cv::Vec3b>(i);
		roiptr = roi.ptr<uchar>(i);
		for (int j = 0; j < mcols; j++)
		{
			if (*(conptr + j) == cv::Vec3b(0,0,255))
			{
				*(roiptr + j) = 255;
			}
			else {
				*(roiptr + j) = 0;
			}
	
		}
	}
	cv::Vec3b pixel(0, 0, 255);
	cv::Point seed(0,0);
	for (int i = 0; i < mrows; i++) {
		conptr = contour.ptr<cv::Vec3b>(i);
		for (int j = 0; j < mcols; j++)
		{
			if (*(conptr + j) == pixel && *(conptr + j + 5)==cv::Vec3b(0,0,0))
			{
				//确定种子点
				/*
				   由于我们是从上往下，从左到右的看点，所以如果看到边界，那么它的右方和下方
				必有非边界点，即0点，故在边界可控范围内挑选种子点
				*/
				seed.x = j + 10;
				seed.y = i + 10;
				break;
			}
		}
		if (seed.x != 0)
		{
			break;
		}
	}
	/*漫水填充，将roi中包含种子点的内的点填充*/
	cv::floodFill(roi, seed, cv::Scalar(255), NULL, cv::Scalar(20));
	cv::Vec3b* imgptr=NULL;

	//得到抠图区域，其他的点全都是白色的
	for (int i = 0; i < mrows; i++)
	{
		roiptr = roi.ptr<uchar>(i);
		imgptr = picure.ptr<cv::Vec3b>(i);
		for (int j = 0; j < mcols; j++)
		{
			if (*(roiptr + j) == 0)
			{
				*(imgptr + j) = cv::Vec3b(255, 255, 255);
			}
		}
	}

	dst = picure.clone();
	return 1;

}



int cutoutimg(cv::Mat& image, cv::Mat& dst,const cv::String& filename)
{
	cv::Mat imagec = image.clone();


	Windows_name = filename.c_str();
	showpicure.release();
	showpicure = imagec.clone();
	picure.release();
	picure = imagec.clone();
	contour.create(imagec.size(), CV_8UC3);
	contour.setTo(0);


	dst.release();
	dst.create(imagec.size(), CV_8UC3);

	cv::namedWindow(Windows_name, cv::WINDOW_AUTOSIZE);
	cv::imshow(Windows_name, imagec);
	while (true)
	{
		cv::setMouseCallback(Windows_name, press, &imagec);

		if (CallBack_flags==0) { 
			cv::destroyAllWindows();
			break; 
		}
		else {
			cv::waitKey(1);
		}
	}
	find_contour(dst);
	cv::destroyAllWindows();
	return 1;
}
