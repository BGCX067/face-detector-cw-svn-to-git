#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "ImageIterator.h"

/////
// Local face measurements structure
// Values are relative to the ROI
typedef struct {
  int xFaceLeft;
  int xFaceRight;
  int xFaceCenter;
  int yEyesTop;
  int yEyesBottom;
  int yEyesCenter;
} FaceMeasurements;

class FaceQualityAssessment
{
public:
	FaceQualityAssessment(cv::Mat &img);
	double poseScore();

private:
	cv::Mat face;
	
	FaceMeasurements measureFace(cv::Mat &gray_image, cv::Mat &x_gradient_image);

	double rotation_score; // known as poseScore in old code.
	double dynamic_range_score;
	double illumination_score;
	double sharpness_score;
	double skin_score;
	double resolution_score;

	double total_score;
};