#include"cutout.hpp"

//��ʾͼ��
cv::Mat showpicure;
//����ͼ��
cv::Mat picure;
//���ı߽�
cv::Mat contour;
//��ˮ����ͼ��
cv::Mat roi;
//��������
std::string Windows_name;
//�ص�������־
int CallBack_flags = 1;


/*
���ص�����

*/
void press(int event, int y, int x, int flags, void* param)
{
	//����
	cv::Vec3b* pixel = NULL;
	cv::Vec3b* picptr = NULL;
	//������ǰ��µ�ʱ��
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
	//��������ͼ��
	cv::Mat canny;
	cv::Canny(picure, canny, value, 2 * value, 3, false);
	////�����㼯
	std::vector<std::vector<cv::Point>> contours;
	////��������
	cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	//
	////��canny����Ӳ����߽��ͼ��
	cv::Mat draw = cv::Mat::zeros(picure.size(), CV_8UC3);
	cv::drawContours(draw, contours, -1, cv::Scalar(0, 0, 255));

	//��ˮͼ
	roi.release();
	roi.create(picure.size(), CV_8UC1);
	uchar* roiptr = NULL;
	int mrows = roi.rows;
	int mcols = roi.cols;
	
	cv::Vec3b* conptr = NULL;
	cv::Vec3b* drawptr = NULL;
	
	//�õ��߽�
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
				//ȷ�����ӵ�
				/*
				   ���������Ǵ������£������ҵĿ��㣬������������߽磬��ô�����ҷ����·�
				���зǱ߽�㣬��0�㣬���ڱ߽�ɿط�Χ����ѡ���ӵ�
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
	/*��ˮ��䣬��roi�а������ӵ���ڵĵ����*/
	cv::floodFill(roi, seed, cv::Scalar(255), NULL, cv::Scalar(20));
	cv::Vec3b* imgptr=NULL;

	//�õ���ͼ���������ĵ�ȫ���ǰ�ɫ��
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
