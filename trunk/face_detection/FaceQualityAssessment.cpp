#include "FaceQualityAssessment.h"
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <Windows.h>

FaceQualityAssessment::FaceQualityAssessment(cv::Mat &img)
{
	face = img;
}

//
FaceMeasurements FaceQualityAssessment::measureFace(cv::Mat &gray_image, cv::Mat &x_gradient_image)
{
	FaceMeasurements fm;
	double max_val;

	double *grad_horiz_proj = new double[gray_image.rows];
	double *grad_vert_proj = new double[gray_image.cols];
	double *eye_location_vert_proj = new double[gray_image.cols];

	for (int i = 0; i < gray_image.rows; i++)
	{
		grad_horiz_proj[i] = 0;
	}

	for (int i = 0; i < gray_image.cols; i++)
	{
		grad_vert_proj[i] = 0;
		eye_location_vert_proj[i] = 0;
	}


	////

	for (int x = 0; x < x_gradient_image.cols; x++)
	{
		for (int y = 0; y < x_gradient_image.rows; y++)
		{
			grad_horiz_proj[y] += x_gradient_image.at<float>(x,y);
			grad_vert_proj[x] += x_gradient_image.at<float>(x,y);
		}
	}


	// 2. find the y-center, where the eyes are located.
	max_val = 0;
	for (int i = 0; i < 2 * x_gradient_image.rows/3; i++)
	{
		if (grad_horiz_proj[i] > max_val)
		{
			max_val = grad_horiz_proj[i];
			fm.yEyesCenter = i;
		}
	}

	fm.yEyesTop = max(0, fm.yEyesCenter - x_gradient_image.rows/10);
	fm.yEyesBottom = min(x_gradient_image.rows - 1, fm.yEyesCenter + x_gradient_image.rows/10);

	// the old code is scrubbing along the x_gradient_image with this IplImageIterator class...but we have restricted the image to only our roi.  is that necessary?

	/*
	// 1. find most probable location for eyes.
	IplImageIterator<unsigned char> grad_it(x_gradient_image, 0, 0, gray_image.size().width - 1, gray_image.size().height - 1);
	while (!grad_it)
	{
		int x = grad_it.column()
	}
	


	*/
	// 3. find the left and right sides of the face.

	// 4. find the center of the face.



	return fm;
}

double FaceQualityAssessment::poseScore()
{
	cv::Mat gray_image;
	cv::Mat x_gradient_image;
	cv::Mat temp;

	cv::cvtColor(face, gray_image, CV_RGB2GRAY);
	cv::Sobel(gray_image, temp, temp.depth(), 1, 0); 
	cv::convertScaleAbs(temp, x_gradient_image, 1, 0); // to be honest, i'm not sure why this is needed...perhaps x_gradient_image can just be the output from cv::Sobel. [TODO]

	// now measure face.
	FaceMeasurements measurements = measureFace(gray_image, x_gradient_image);

	cv::line(x_gradient_image, cv::Point(0, measurements.yEyesTop), cv::Point(x_gradient_image.cols, measurements.yEyesTop), cv::Scalar(100, 200, 50, 100), 3);
	cv::line(x_gradient_image, cv::Point(0, measurements.yEyesBottom), cv::Point(x_gradient_image.cols, measurements.yEyesBottom), cv::Scalar(100, 200, 50, 100), 3);
	cv::line(x_gradient_image, cv::Point(0, measurements.yEyesCenter), cv::Point(x_gradient_image.cols, measurements.yEyesCenter), cv::Scalar(255, 255, 255, 255), 3);
	cv::namedWindow("Sobel image.");
	cv::imshow("Sobel image.", x_gradient_image);

	cv::namedWindow("Temp image.");
	cv::imshow("Temp image.", temp);
	cv::waitKey();

	/*
	for (int i = 0; i < temp.size().width; i++)
	{
		for (int j = 0; j < temp.size().height; j++)
		{
			temp.at<unsigned char>(i, j) = 100;
			//temp.at<unsigned char>(i + 1, j) = 100;
			//temp.at<unsigned char>(i, j + 1) = 100;
			std::cout << "(" << i << ", " << j << ")" << std::endl;
			cv::imshow("Temp image.", temp);
			cv::waitKey();
		}
	}
	*/

	return 0;
}