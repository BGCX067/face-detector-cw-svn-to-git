#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

class Detector
{
public:
	Detector();
	void processFrame(cv::Mat &img, cv::Mat &out);
	
	
private:
	int x;
	void canny(cv::Mat &img, cv::Mat &out);
};

#endif