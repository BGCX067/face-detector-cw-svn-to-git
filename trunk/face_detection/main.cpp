#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "VideoProcessor.h"
#include "FaceQualityAssessment.h"
#include "FragmentsTracker.h"

#include <stdio.h>
#include <iostream>
#include <sstream>

// these shouldn't be global when done.
///////////////////////////////////////////////////////
cv::CascadeClassifier face_cascade;
cv::RNG rng(12345);
bool face_detected;
cv::Mat face_template;
FragmentsTracker *tracker;



void face_detect_1(cv::Mat &img, cv::Mat &out)
{
	if (!face_detected)
	{
		// now do the detection...
		std::vector<cv::Rect> faces;
		//cv::Mat gray_frame;
	
		cv::cvtColor(img, out, CV_BGR2GRAY);
		cv::equalizeHist(out, out);

		face_cascade.detectMultiScale(out, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

		if (faces.size() > 0)
		{
			// copy template.
			face_template = out(cv::Range(faces[0].y, faces[0].y + faces[0].height), cv::Range(faces[0].x,faces[0].x + faces[0].width));
			face_detected = true;
			tracker = new FragmentsTracker(img, face_template, faces[0]);
			cv::namedWindow("Tracker");
			cv::imshow("Tracker", img);
		}
	}
	else
	{
		// keep tracking.
		tracker->handleFrame(img);
		// do stuff.
		cv::namedWindow("Template");
		cv::imshow("Template", face_template);

		cv::imshow("Tracker", img);
		//cv::waitKey();
	}



	//fqa stuff.
	/*

	for (int i = 0; i < faces.size(); i++)
	{
		//cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
		//cv::ellipse(img, center, cv::Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);
		cv::Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
	    ellipse( out, center, cv::Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );
		std::stringstream ss;
		ss << "width = " << faces[i].width << " height = " << faces[i].height << std::endl;
		cv::putText(out, ss.str(), cv::Point(0,15), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 100));

		// now... take a ROI on the face, calculate rotation score, smallest range of gray-level intensities score, etc.
		cv::Mat roi = img(faces[i]);
		out = roi;
		//cv::set


		/////// now we will try to get the rotation score...
		// just some temp work for now.
		FaceQualityAssessment fqa(out);
		double tmp = fqa.poseScore();
	}
	*/


}

int main()
{
	printf("Testing!!\n");
	VideoProcessor processor;
	face_detected = false;

	// open file.
	if (!processor.setInput("C:/face.avi")) {std::cout << "Could not load sample video..." << std::endl; cv::waitKey(); return -1;}
	processor.displayInput("Input Video");
	processor.displayOutput("Output Video");

	//play video at original frame rate
	processor.setDelay(1000./processor.getFrameRate());
	processor.setFrameProcessor(face_detect_1);

	////////////////////////////////////////////////////////////////
	// load and check training xml here.
	std::string face_cascade_name = "C:/haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) {printf("--(!)Error loading\n"); cv::waitKey(); return -1; }

	processor.run();
	cv::waitKey();
	
}