#ifndef CUTOUT_HPP
#define CUTOUT_HPP

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<vector>
#include<string>



/*

*/
void press(int event, int y, int x, int flag, void* param);
/*

*/
int find_contour(cv::Mat& dst, int value = 100);


/*
*/
int cutoutimg(cv::Mat& image, cv::Mat& dst,const cv::String& filename);

#endif // !CUTOUT_HPP
