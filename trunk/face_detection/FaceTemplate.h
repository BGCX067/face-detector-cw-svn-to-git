#ifndef FACETEMPLATE_H
#define FACETEMPLATE_H
#include <opencv2\core\core.hpp>

class FaceTemplate
{
public:
	FaceTemplate(cv::Mat t, int _x, int _y, int h, int w);
	cv::Mat getTemplate();
	void setTemplate(cv::Mat t);
	int getWidth() const;
	int getHeight() const;
	int getYTopLeft() const;
	int getXTopLeft() const;
	int getYCenter() const;
	int getXCenter() const;

	void setY(int _y);
	void setX(int _x);
	void setHeight(int h);
	void setWidth(int w);
	
private:
	cv::Mat face_template;
	int height;
	int width;
	int x; // from topleft
	int y; // from topleft.
};
#endif