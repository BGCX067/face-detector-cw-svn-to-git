#include "Detector.h"

Detector::Detector()
{
	x = 5;
}

void Detector::canny(cv::Mat &img, cv::Mat &out)
{
	//convert to gray
	if (img.channels() == 3)
		cv::cvtColor(img, out, CV_BGR2GRAY);

	// compute canny edges.
	cv::Canny(out, out, 100, 200);

	// invert the image.
	cv::threshold(out, out, 128, 255, cv::THRESH_BINARY_INV);
}


void Detector::processFrame(cv::Mat &img, cv::Mat &out)
{
	canny(img, out);
}