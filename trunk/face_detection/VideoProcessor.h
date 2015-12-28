#ifndef VIDEOPROCESSOR_H
#define VIDEOPROCESSOR_H

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

class FrameProcessor
{
	public:
		// processing method..
		virtual void process(cv::Mat &input, cv::Mat &output) = 0;
};

class VideoProcessor
{
public:
	VideoProcessor() : call_it(true), delay(0), fnumber(0), stop(false), frame_to_stop(-1) {}

	// setting the callback ftn that will be called for each frame.
	// takes as a parameter: a function whose params are two cv::Mats.
	void setFrameProcessor(void (*frameProcessingCallback) (cv::Mat&, cv::Mat&))
	{
		// invalidate frame processor class instance
		frame_processor = 0;
		// this is the frame processor ftn that will be called
		process = frameProcessingCallback;
		callProcess();
	}

	void setFrameProcessor(FrameProcessor *frame_processor_ptr)
	{
		// invalidate the callback ftn
		process = 0;
		// this is the frame processor isntance that will be called
		frame_processor = frame_processor_ptr;
		callProcess();
	}

	// set name of the video file.
	bool setInput(std::string filename)
	{
		fnumber = 0;
		// in case a resource was already associated with the videocapture instance..
		capture.release();
		images.clear();

		// open the video file.
		return capture.open(filename);
	}

	bool setInput(const std::vector<std::string> &imgs)
	{
		fnumber = 0;
		capture.release();

		images = imgs;
		it_img = images.begin();

		return true;
	}

	void displayInput(std::string wn)
	{
		window_name_input = wn;
		cv::namedWindow(window_name_input);
	}

	void displayOutput(std::string wn)
	{
		window_name_output = wn;
		cv::namedWindow(window_name_output);
	}

	void dontDisplay()
	{
		cv::destroyWindow(window_name_input);
		cv::destroyWindow(window_name_output);
		window_name_input.clear();
		window_name_output.clear();
	}

	bool isOpened()
	{
		return capture.isOpened() || !images.empty();
	}

	bool isStopped()
	{
		return stop;
	}

	void stopIt()
	{
		stop = true;
	}

	// set a delay between each frame.  0 means wait at each frame.  negative means no delay.
	void setDelay(int d)
	{
		delay = d;
	}

	void callProcess()
	{
		call_it = true;
	}

	void dontCallProcess()
	{
		call_it = false;
	}

	void stopAtFrameNo(long frame)
	{
		frame_to_stop = frame;
	}

	long getFrameNumber()
	{
		long fnumber = static_cast<long>(capture.get(CV_CAP_PROP_POS_FRAMES));
		return fnumber;
	}

	double getFrameRate()
	{
		// undefined for vector of images.
		if (images.size() != 0) return 0;

		double r = capture.get(CV_CAP_PROP_FPS);
		return r;
	}


	// ****main method****
	// to grab (and process) the frames of the sequence.
	void run()
	{
		// current frame.
		cv::Mat frame;
		// output frame
		cv::Mat output;

		// if no capture dev has been set...
		if (!isOpened())
			return;

		stop = false;

		while (!isStopped())
		{
			// read next frame, if any.
			if (!readNextFrame(frame))
				break;

			// display input frame.
			if (window_name_input.length() != 0)
				cv::imshow(window_name_input, frame);

			// calling the process ftn.
			if (call_it)
			{
				if (process) // if callback ftn
				{
					process(frame, output);
				}
				else if (frame_processor) //if class interface instance
				{
					frame_processor->process(frame, output);
				}
				fnumber++;
			}
			else
			{
				output = frame;
			}

			if (window_name_output.length() != 0)
			{
				cv::imshow(window_name_output, output);
			}

			// introduce a delay.
			if (delay >= 0 && cv::waitKey(delay) >= 0)
				stopIt();

			if (frame_to_stop >= 0 && getFrameNumber() == frame_to_stop)
				stopIt();
		}
	}

private:
	// opencv video capture object
	cv::VideoCapture capture;

	// Pointer to the class implementing the FrameProcessor interface
	FrameProcessor *frame_processor;

	// callback function to be called for the processing of each frame
	void (*process)(cv::Mat&, cv::Mat&);

	// bool to determine if the process callback will be called
	bool call_it;

	// input display window name
	std::string window_name_input;

	// output display window name
	std::string window_name_output;

	// delay between each processing
	int delay;

	// number of processed frames
	long fnumber;

	// stop at this frame number
	long frame_to_stop;

	// to stop the processing
	bool stop;

	// vector of image fileanme to be used as input
	std::vector<std::string> images;
	// image vector iterator
	std::vector<std::string>::const_iterator it_img;

	bool readNextFrame(cv::Mat &frame)
	{
		if (images.size() == 0)
			return capture.read(frame);
		else
		{
			if (it_img != images.end())
			{
				frame = cv::imread(*it_img);
				it_img++;
				return frame.data != 0;
			}
			else
			{
				return false;
			}
		}
	}
};

#endif