#include "FaceTemplate.h"

FaceTemplate::FaceTemplate(cv::Mat t, int _x, int _y, int h, int w)
{
	face_template = t;
	x = _x;
	y = _y;
	height = h;
	width = w;
}

cv::Mat FaceTemplate::getTemplate()
{
	return face_template;
}

void FaceTemplate::setTemplate(cv::Mat t)
{
	face_template = t;
}

int FaceTemplate::getWidth() const
{
	return width;
}

int FaceTemplate::getHeight() const
{
	return height;
}

int FaceTemplate::getXTopLeft() const
{
	return x;
}

int FaceTemplate::getYTopLeft() const
{
	return y;
}

int FaceTemplate::getXCenter() const
{
	int half_w = (int) floor((double)width / 2.0);
	return x + half_w;
}

int FaceTemplate::getYCenter() const
{
	int half_h = (int) floor((double)height / 2.0);
	return y + half_h;
}

void FaceTemplate::setY(int _y)
{
	y = _y;
}

void FaceTemplate::setX(int _x)
{
	x = _x;
}

void FaceTemplate::setHeight(int h)
{
	height = h;
}

void FaceTemplate::setWidth(int w)
{
	width = w;
}