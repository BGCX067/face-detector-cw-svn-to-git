#include "FragmentsTracker.h"
#include <iostream>

FragmentsTracker::FragmentsTracker(cv::Mat &img, cv::Mat t, cv::Rect r) : ft(t, r.x, r.y, r.height, r.width)
{
	// initialize the tracker.

	// allocate space for IIV_I and combined vote maps?
	cv::Mat *cur_ii;

	for (int i = 0; i < num_bins; i++)
	{
		cur_ii = new cv::Mat(img.rows, img.cols, CV_32S);
		IIV_I.push_back(cur_ii);
	}

	// template center
	curr_pos_y = ft.getYCenter();
	curr_pos_x = ft.getXCenter();

	buildTemplatePatchHistogram(&t, template_patch_histograms);
	drawRectangle(ft.getYTopLeft(), ft.getXTopLeft(), ft.getHeight(), ft.getWidth(), img);

}


void FragmentsTracker::handleFrame(cv::Mat &img)
{
	// build this image's IH.  Only work with this data structure, not the image.
	int img_height, img_width;
	
	computeIntegralHistogram(&img, IIV_I);
	img_height = img.rows;
	img_width = img.cols;

	// find the current template in the current image.
	vector<int> x_coords;
	vector<int> y_coords;

	int new_yM, new_xM;
	double score_M;

	// returns located position of face in frame.
	findTemplate(template_patch_histograms, patches, img_height, img_width,
		curr_pos_y - search_margin, curr_pos_x - search_margin, 
		curr_pos_y + search_margin, curr_pos_x + search_margin, 
		new_yM, new_xM, score_M, x_coords, y_coords);
	// [TODO]

	printf("Just chillin. \n");
	// update template.

	updateTemplate(ft.getHeight(), ft.getWidth(), new_yM, new_xM, 1, img);
	drawRectangle(ft.getYTopLeft(), ft.getXTopLeft(), ft.getHeight(), ft.getWidth(), img);
}

void FragmentsTracker::updateTemplate(int new_height,int new_width, int new_center_y, int new_center_x, double scale_factor, cv::Mat &img)
{
	int t_halfw = (int) floor((double)new_width / 2.0);
	int t_halfh = (int) floor((double)new_height / 2.0);

	ft.setY(new_center_y - t_halfh);
	ft.setX(new_center_x - t_halfw);

	ft.setHeight(new_height);
	ft.setWidth(new_width);

	// make sure we stay inside the image (without changing template size).
	if (ft.getYTopLeft() < 0) {ft.setY(0);}
	if (ft.getXTopLeft() < 0) {ft.setX(0);}
	if (ft.getYTopLeft() > img.rows - new_height)
	{
		ft.setY(img.rows - new_height);
	}
	if (ft.getXTopLeft() > img.cols - new_width)
	{
		ft.setX(img.cols - new_width);
	}
}

void FragmentsTracker::findTemplate(vector<vector<double>*> &template_histograms, vector<Patch*> &image_patches, int img_h, int img_w, int minrow, int mincol, int maxrow, int maxcol,
		int &result_y, int &result_x, double &score, vector<int> &x_coords, vector<int> &y_coords)
{
	if (minrow < 0) minrow = 0;
	if (mincol < 0) mincol = 0;
	if (maxrow >= img_h) maxrow = img_h - 1;
	if (maxcol >= img_w) maxcol = img_w - 1;

	// in handle frame we already computed the IH, we have updated it in IIV_I
	std::cout << "In findTemplate()" << std::endl;
	std::cout << "------------------------------------" << std::endl << std::endl;
	std::cout << "minrow = " << minrow << " and maxrow = " << maxrow << std::endl;
	std::cout << "mincol = " << mincol << " and maxcol = " << maxcol << std::endl;

	vector<double> p_scores;
	cv::Mat *combined_vote = new cv::Mat(maxrow - minrow + 1, maxcol - mincol + 1, CV_32F);
	computeAllPatchVotes(template_histograms, image_patches, img_h, img_w, minrow, mincol, maxrow, maxcol, combined_vote, x_coords, y_coords, p_scores);

	cv::Point min_loc, max_loc;
	double min_val, max_val;

	cv::minMaxLoc(*combined_vote, &min_val, &max_val, &min_loc, &max_loc);

	int cx = min_loc.x; 
	int cy = min_loc.y;

	result_y = cy + minrow;
	result_x = cx + mincol;
	score = min_val;
}

// runs on all patches and computes each one's vote map.  then combines all the vote maps to a single vote map
void FragmentsTracker::computeAllPatchVotes(vector<vector<double>*> &patch_histograms, vector<Patch*> &tested_patches, int img_h, int img_w, int minrow, int mincol, int maxrow, int maxcol,
		cv::Mat *combined_vote, vector<int> &x_coords, vector<int> &y_coords, vector<double> &patch_scores)
{
	vector<Patch *>::iterator it;
	cv::Mat *curr_vm;
	vector<double> *curr_patch_histogram;

	patch_vote_maps.clear();
	x_coords.clear();
	y_coords.clear();
	patch_scores.clear();

	vector<int> vote_regions_minrow;
	vector<int> vote_regions_mincol;
	vector<int> vote_regions_maxrow;
	vector<int> vote_regions_maxcol;
	
	int minx, miny, maxx, maxy;
	int vm_width = maxcol - mincol + 1;
	int vm_height = maxrow - minrow + 1;
	int i = 0;

	// pass on every patch and build its vote map.
	// allocate vote map matrices in this loop..
	for (it = tested_patches.begin(); it != tested_patches.end(); it++)
	{
		curr_vm = new cv::Mat();
		*curr_vm = cv::Mat::zeros(vm_height, vm_width, CV_32F);
		// compute current patch histogram
		// std::cout << "vm rows = " << vm_height << " and vm cols = " << vm_width << std::endl;

		curr_patch_histogram = patch_histograms[i];

		computeSinglePatchVotes((*it), *curr_patch_histogram, minrow, mincol, maxrow, maxcol, curr_vm, miny, minx, maxy, maxx);
		
		patch_vote_maps.push_back(curr_vm);
		vote_regions_minrow.push_back(miny);
		vote_regions_mincol.push_back(minx);
		vote_regions_maxrow.push_back(maxy);
		vote_regions_maxcol.push_back(maxx);

		// find the position based on this patch:

		cv::Point min_loc, max_loc;
		double minval, maxval;

		cv::minMaxLoc(*curr_vm, &minval, &maxval, &min_loc, &max_loc);

		x_coords.push_back(mincol + min_loc.x);
		y_coords.push_back(minrow + min_loc.y);
		patch_scores.push_back(minval);
		i++;
	}

	// combine patch votes - using a a quantile based score makes this combination robust to occlusions.
	combineVoteMapsMedian(patch_vote_maps, combined_vote);

	// release stuff.

	vector<cv::Mat *>::iterator vm_it;
	for (vm_it = patch_vote_maps.begin(); vm_it != patch_vote_maps.end(); vm_it++)
	{
		(*vm_it)->release();
	}
	patch_vote_maps.clear();
}

// at each hypothesis, sorts the score given by each patch and tkaes the 'q'th quantile as the score.
// This ignores outlier scores contributed by patches affected by occlusions, for example.

void FragmentsTracker::combineVoteMapsMedian(vector<cv::Mat *> &vote_maps, cv::Mat *v)
{
	int rows = (vote_maps[0])->rows;
	int cols = (vote_maps[0])->cols;
	int size = vote_maps.size();

	vector<double> fv;

	int q_index = (int) floor(((double) size) / 4.0);

	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			// take all the values this pixel got in all the vote maps and compute their median.
			fv.clear();

			for (int p = 0; p < size; p++)
			{
				double z = vote_maps[p]->at<float>(row, col);
				fv.push_back(z);
			}

			std::sort(fv.begin(), fv.end());

			double q = fv[q_index];
			v->at<float>(row, col) = q;
		}
	}
}

void FragmentsTracker::printMatrix(cv::Mat &m)
{
	for (int row = 0; row < m.rows; row++)
	{
		for (int col = 0; col < m.cols; col++)
		{
			std::cout << m.at<float>(row, col) << ", ";
		}
		std::cout << endl;
	}
}
void FragmentsTracker::computeSinglePatchVotes (Patch *p , vector<double> &hist,
									int minrow, int mincol,
									int maxrow, int maxcol,
									cv::Mat *votes, int &min_r, int &min_c,
									int &max_r, int &max_c)
{
	int M = (*(IIV_I.begin()))->rows;
	int N = (*(IIV_I.begin()))->cols;
	int minx, maxx, miny, maxy;

	// compute left margin.
	if (p->w > p->dx)
	{
		minx = p->w;
	}
	else
	{
		minx = p->dx;
	}

	// compute right margin.
	if (p->dx < -p->w)
	{
		maxx = N - 1 + p->dx;
	}
	else
	{
		maxx = N - 1 - p->w;
	}

	// compute up margin.
	if (p->h > p->dy)
	{
		miny = p->h;
	}
	else
	{
		miny = p->dy;
	}

	// compute bottom margin
	if (p->dy < -p->h)
	{
		maxy = M - 1 + p->dy;
	}
	else
	{
		maxy = M - 1 - p->h;
	}

	// patch center (y,x) votes for the target center (y - dy, x - dx)
	// we want votes only in the range min/max - row/col
	// we now enforce that:

	if (miny < minrow + p->dy) miny = minrow + p->dy;
	if (maxy > maxrow + p->dy) maxy = maxrow + p->dy;
	if (minx < mincol + p->dx) minx = mincol + p->dx;
	if (maxx > maxcol + p->dx) maxx = maxcol + p->dx;

	//cvSet(votes, cvScalar(1000.0));
	int col, row;
	double z = 0;
	double sum_z = 0;
	double t = 0;
	vector<double> curr_hist;

	//printf("Printing votes now... \n");
	//printMatrix(*votes);
	//printf("Done printing votes.\n");
	for (col = minx; col <= maxx; col++)
	{
		for (row = miny; row <= maxy; row++)
		{
			computeHistogram(row - p->h, col - p->w, row + p->h, col + p->w, IIV_I, curr_hist);

			// compare the two histograms.
			z = compareHistograms(hist, curr_hist);

			// now the votemap is not the whole image, but only the portion between min - max, row - col
			// so, y - dy = minrow --> vote for index = 0
			//std::cout << "going to assign " << z << " to row, col pair " << row << ", " << col << std::endl;
			votes->at<float>(row - p->dy - minrow, col - p->dx - mincol) = z;
		}
	}

	// return the region where votes were added..
	min_c = minx - p->dx;
	max_c = maxx - p->dx;
	min_r = miny - p->dy;
	max_r = maxy - p->dy;
}

// Kolmogorov - Smirnov distance.  Equivalent to earth mover's distance, but faster for one dimensional data..
double FragmentsTracker::compareHistograms(vector<double> &h1, vector<double> &h2)
{
	double sum = 0;
	double cdf1 = 0;
	double cdf2 = 0;
	double z = 0;
	double ctr = 0;
	vector <double>::iterator it1, it2;

	for (it1 = h1.begin(), it2 = h2.begin(); it1 != h1.end(), it2 != h2.end(); it1++, it2++)
	{
		cdf1 += (*it1);
		cdf2 += (*it2);

		z = cdf1 - cdf2;
		sum += abs(z);
		ctr++;
	}
	return (sum/ctr);
}

void FragmentsTracker::drawRectangle(int tl_y, int tl_x, int height, int width, cv::Mat &img)
{
	cv::Point tl;
	tl.y = tl_y;
	tl.x = tl_x;

	cv::Point br;
	br.y = tl_y + height - 1;
	br.x = tl_x + width - 1;

	cv::rectangle(img, tl, br, cv::Scalar(255,0,0),3);
}

void FragmentsTracker::buildTemplatePatchHistogram(cv::Mat *t, vector< vector<double>* > &patch_histograms)
{
	// clear the current histograms.

	vector< vector<double>* >::iterator it;
	for (it = patch_histograms.begin(); it != patch_histograms.end(); it++)
	{
		(*it)->clear();
		delete (*it);
	}
	patch_histograms.clear();


	// define the patches on this template...
	definePatches(t->rows, t->cols, patches);

	// compute the integral histogram on the template.
	for (int i = 0; i < IIV_T.size(); i++)
	{
		IIV_T[i]->release();
	}
	IIV_T.clear();

	cv::Mat *curr_ii;
	for (int i = 0; i < num_bins; i++)
	{
		curr_ii = new cv::Mat(t->rows, t->cols, CV_32F);
		IIV_T.push_back(curr_ii);
	}
	computeIntegralHistogram(t, IIV_T);

	// now compute histograms for every defined patch.

	vector<double>* curr_histogram;
	int t_cx = (int) floor((double)t->cols / 2.0);
	int t_cy = (int) floor((double)t->rows / 2.0);

	vector<Patch*>::iterator it2;
	int ctr = 0;
	for (it2 = patches.begin(); it2 != patches.end(); it2++)
	{
		// compute current patch histogram.

		int p_cx = t_cx + (*it2)->dx;
		int p_cy = t_cy + (*it2)->dy;

		curr_histogram = new vector<double>;
		computeHistogram(p_cy - (*it2)->h, p_cx - (*it2)->w, p_cy + (*it2)->h, p_cx + (*it2)->w, IIV_T, *curr_histogram);

		patch_histograms.push_back(curr_histogram);
		ctr++;
	}
}

// use the integral histogram to quickly compute a histogram in a rectangular region.
void FragmentsTracker::computeHistogram(int topleft_y, int topleft_x, int bottomright_y, int bottomright_x, vector<cv::Mat *> &iiv, vector<double> &hist)
{
	vector<cv::Mat *>::iterator it;
	hist.clear();
	double left, up, diag;
	double sum = 0;
	double z;

	for (it = iiv.begin(); it != iiv.end(); it++)
	{
		if (topleft_x == 0)
		{
			left = 0;
			diag = 0;
		}
		else
		{
			left = (*it)->at<float>(bottomright_y, topleft_x - 1); // this is strange........ might be weird adapting from getreal2d.	
		}

		if (topleft_y == 0)
		{
			up = 0;
			diag = 0;
		}
		else
		{
			up = (*it)->at<float>(topleft_y - 1, bottomright_x);
		}

		if (topleft_x > 0 && topleft_y > 0)
		{
			diag = (*it)->at<float>(topleft_y - 1, topleft_x - 1);
		}

		z = (*it)->at<float>(bottomright_y, bottomright_x) - left - up + diag;
		hist.push_back(z);
		sum += z;
	}

	vector <double>::iterator it2;
	for (it2 = hist.begin(); it2 != hist.end(); it2++)
	{
		(*it2) /= sum; // is this just normalizing?
	}
}

// accepts the template size and returns a vector of patches on this window size.
// patches are 20 vertical strips, each of height half-template-height and 20 similar horizontal strips.
void FragmentsTracker::definePatches(int height, int width, vector <Patch*> &patch_vec)
{
	for (int i = 0; i < patch_vec.size(); i++)
	{
		delete patch_vec[i];
	}

	patch_vec.clear();

	// 10 vertical strips and 10 horizontal strips.  each strip is divided into 2.
	int pw_f = (int) floor( ((double)width) * 0.25);
	int ph_f = (int) floor( ((double)height) * 0.25);

	int pw_s = (int) floor( ((double)width) * 0.05 + 0.5);
	int ph_s = (int) floor( ((double)height) * 0.05 + 0.5);

	if (pw_f < 1) pw_f = 1;
	if (ph_f < 1) ph_f = 1;
	if (pw_s < 1) pw_s = 1;
	if (ph_s < 1) ph_s = 1;

	// template center.
	int t_cx = (int) floor( ((double)width) / 2.0);
	int t_cy = (int) floor( ((double)height) / 2.0);

	int dx = (int) floor( ((double)width) / 4.0);
	int dy = (int) floor( ((double)height) / 4.0);

	if (dx > t_cx - pw_f) dx = t_cx - pw_f;
	if (dx > width - 1 - pw_f - t_cx) dx = width - 1 - pw_f - t_cx;
	if (dy > t_cy - ph_f) dy = t_cy - ph_f;
	if (dy > height - 1 - t_cy - ph_f) dy = height - 1 - t_cy - ph_f;

	Patch *p;

	// horizontal patches.
	for (int i = ph_s; i <= height - 1 - ph_s; i = i + 2 * ph_s)
	{
		p = new Patch;
		p->dy = i - t_cy;
		p->dx = -dx;
		p->h = ph_s;
		p->w = pw_f;
		patch_vec.push_back(p);

		p = new Patch;
		p->dy = i - t_cy;
		p->dx = dx;
		p->h = ph_s;
		p->w = pw_f;
		patch_vec.push_back(p);
	}

	// vertical patches.
	for (int i = pw_s; i <= width - 1 - pw_s; i = i + 2 * pw_s)
	{
		p = new Patch;
		p->dx = i - t_cx;
		p->dy = -dy;
		p->w = pw_s;
		p->h = ph_f;
		patch_vec.push_back(p);

		p = new Patch;
		p->dx = i - t_cx;
		p->dy = dy;
		p->w = pw_s;
		p->h = ph_f;
		patch_vec.push_back(p);
	}

}

void FragmentsTracker::getPixelBin(cv::Mat *img, cv::Mat *bin_mat)
{
	double bin_width = floor(256. / (double)(num_bins));
	int dims = bin_mat->dims;
	bool d = bin_mat->data;

	for (int row = 0; row < bin_mat->rows; row++)
	{
		for (int col = 0; col < bin_mat->cols; col++)
		{
			int b = (int)(floor (img->at<unsigned char>(row, col) / bin_width));
			if (b > num_bins - 1)
			{
				b = num_bins - 1;
			}

			bin_mat->at<float>(row, col) = b;
		}
	}
	return;
}

void FragmentsTracker::computeIntegralHistogram(cv::Mat *img, vector<cv::Mat *> &vector_integral_image)
{
	// reset iiv matrices.
	vector<cv::Mat *>::iterator it;
	for (it = vector_integral_image.begin(); it != vector_integral_image.end(); it++)
	{
		(*(*it)) = cv::Mat::zeros((*it)->rows, (*it)->cols, (*it)->type());
	}

	cv::Mat *curr_bin_mat = new cv::Mat(img->rows, img->cols, CV_32F);
	getPixelBin(img, curr_bin_mat);


	// compute integral histogram now.
	int row, col, count, curr_bin;
	double v_up, v_left, v_diag, z;
	for (row = 0; row < img->rows; row++)
	{
		for (col = 0; col < img->cols; col++)
		{
			curr_bin = curr_bin_mat->at<float>(row, col);

			for (it = vector_integral_image.begin(), count = 0; it != vector_integral_image.end(); it++, count++)
			{
				// no up
				if (row == 0)
				{
					v_up = 0;
					v_diag = 0;
				}
				else
				{
					v_up = (*it)->at<float>(row - 1, col);
				}

				// no left
				if (col == 0)
				{
					v_left = 0;
					v_diag = 0;
				}
				else
				{
					v_left = (*it)->at<float>(row, col - 1);
				}

				// diag exists.
				if (row > 0 && col > 0)
				{
					v_diag = (*it)->at<float>(row - 1, col - 1);
				}

				// set cell value.
				z = v_left + v_up - v_diag;
				if (curr_bin == count)
				{
					z++;
				}
				(*it)->at<float>(row, col) = z;
			}
		}
	}
	curr_bin_mat->release();
}