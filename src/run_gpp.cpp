#include "general_img.h"
#include "sr_gpp.h"

using namespace std;
using namespace dlib;

Mat run_gpp(Mat img_y, int sf, double sigma, double beta)
{
	Mat img_bb;

	img_y.convertTo(img_y, CV_64FC1);
	cv::resize(img_y, img_bb, Size(img_y.size()*sf),0,0, INTER_LINEAR);
	std::vector<Mat> Grad_bb;
	ComputeGrad(img_bb, Grad_bb);
	std::vector<dlib::matrix<double>> Grad_High;
	Grad_High = grad_gpp(Grad_bb, sf);

	//cout << "sigam array " << subm(Grad_High[0], range(0,0), range(0,10)) << endl;

	Mat img_hr;
	img_hr = generalImgFromGpp(img_y, Grad_High, 2, sigma, beta);
	return img_hr;
}

