#ifndef FRAGMENTSTRACKER_H
#define FRAGMENTSTRACKER_H
#include <opencv2\core\core.hpp>
#include "FaceTemplate.h"

using namespace std;
// Patch.  a rectangular patch to be tracked.
struct Patch
{
	int dx; // of patch center with respect to template center
	int dy; // of patch center with respect to template center
	int w; // patch width is 2*w + 1
	int h; // patch height is 2*h + 1
};


// Parameters for initializing the tracker.
struct Parameters
{
	// initial target template...
	int inital_tl_y;
	int initial_tl_x;
	int initial_br_y;
	int initial_br_x;

	// search area around position in previous frame.
	int search_margin;

	// number of bins in histogram.
	int b;

	// metric used for histogram comparison.
	int metric_used;
};



// This is the tracker object.

class FragmentsTracker
{
private:
	//
	// variables
	//

	FaceTemplate ft;
	vector <cv::Mat *> IIV_I;
	vector <cv::Mat *> IIV_T;
	
	vector <Patch*> patches;
	vector<vector<double>*> template_patch_histograms;
	vector<cv::Mat *> patch_vote_maps;

	int curr_pos_y;
	int curr_pos_x;
	/*int curr_template_tl_y;
	int curr_template_tl_x;
	int curr_template_height;
	int curr_template_width;*/ // these are replaced by FaceTemplate object.

	static const int num_bins = 16; // number of bins for histograms.
	static const int search_margin = 7;


	//
	// functions
	//

	void computeIntegralHistogram(cv::Mat *img, vector<cv::Mat *> &vector_integral_image);
	void getPixelBin(cv::Mat *img, cv::Mat *bin_mat);
	void buildTemplatePatchHistogram(cv::Mat *t, vector< vector<double>* > &patch_histograms);
	void definePatches(int height, int width, vector <Patch*> &patch_vec);
	void computeHistogram(int topleft_y, int topleft_x, int bottomright_y, int bottomright_x, vector<cv::Mat *> &iiv, vector<double> &hist);
	void drawRectangle(int tl_y, int tl_x, int height, int width, cv::Mat &img);

	void findTemplate(vector<vector<double>*> &template_histograms, vector<Patch*> &image_patches, int img_h, int img_w, int minrow, int mincol, int maxrow, int maxcol,
		int &result_y, int &result_x, double &score, vector<int> &x_coords, vector<int> &y_coords);

	void computeAllPatchVotes(vector<vector<double>*> &patch_histograms, vector<Patch*> &tested_patches, int img_h, int img_w, int minrow, int mincol, int maxrow, int maxcol,
		cv::Mat *combined_vote, vector<int> &x_coords, vector<int> &y_coords, vector<double> &patch_scores);

	void computeSinglePatchVotes (Patch *p , vector<double> &hist,
										int minrow, int mincol,
										int maxrow, int maxcol,
										cv::Mat *votes, int &min_r, int &min_c,
										int &max_r, int &max_c);
	void combineVoteMapsMedian(vector<cv::Mat *> &vote_maps, cv::Mat *v);
	double compareHistograms(vector<double> &h1, vector<double> &h2);

	void printMatrix(cv::Mat &m);

	void updateTemplate(int new_height,int new_width, int new_cy, int new_cx, double scale_factor, cv::Mat &img);


public:
	FragmentsTracker(cv::Mat &img, cv::Mat t, cv::Rect r);
	void handleFrame(cv::Mat &img);
};

#endif