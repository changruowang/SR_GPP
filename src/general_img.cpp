#include <dlib/matrix.h>
#include "general_img.h"

using namespace dlib;
using namespace cv;
using namespace std;

void converMatrix2Mat(matrix<double> m, Mat& out)
{
	out.create(m.nr(), m.nc(), CV_64FC1);
	for(int r=0; r < m.nr(); r++)
		for (int c = 0; c < m.nc(); c++)
			out.at<double>(r, c) = m(r, c);
}

void ComputeDivergence(std::vector<Mat> grad, Mat& out)
{
	if (grad.size() != 2)
		cout << "size   erro !!!!!!!!" << endl;

	double k1[1][3] = { -0.5,0,0.5 };  //水平方向的核
	double k2[3][1] = { -0.5,0,0.5 };  //垂直的核
	Mat div_x, div_y;

	Mat Kore = Mat(1, 3, CV_64FC1, k1);
	Mat Kore2 = Mat(3, 1, CV_64FC1, k2);

	filter2D(grad[0], div_x, -1, Kore, Point(-1, -1), 0, BORDER_REFLECT);
	filter2D(grad[1], div_y, -1, Kore2, Point(-1, -1), 0, BORDER_REFLECT);

	out = div_x + div_y;
}

void myGaussianBlur(Mat& input, Mat& result, int size)
{
	if (size == 6)
	{
		double guass_kernel[6][6] = { 1.27970158549470e-08,	3.31021215991598e-06,	5.32389388435179e-05,	5.32389388435179e-05,	3.31021215991598e-06,	1.27970158549470e-08,
									3.31021215991598e-06,	0.000856254666545540,	0.0137713498786289,	0.0137713498786289,	0.000856254666545540,	3.31021215991598e-06,
									5.32389388435179e-05,	0.0137713498786289,	0.221487934477174,	0.221487934477174,	0.0137713498786289,	5.32389388435179e-05,
									5.32389388435179e-05,	0.0137713498786289,	0.221487934477174,	0.221487934477174,	0.0137713498786289,	5.32389388435179e-05,
									3.31021215991598e-06,	0.000856254666545540,	0.0137713498786289,	0.0137713498786289,	0.000856254666545540,	3.31021215991598e-06,
									1.27970158549470e-08,	3.31021215991598e-06,	5.32389388435179e-05,	5.32389388435179e-05,	3.31021215991598e-06,	1.27970158549470e-08 };
		Mat Kore = Mat(6, 6, CV_64FC1, guass_kernel);
		filter2D(input, result, -1, Kore, Point(2, 2), 0, BORDER_REPLICATE);
	}

}

void GenerateLRImage_GaussianKernel(Mat hrimg, Mat& lrimg,int s, double sigma)
{
	if (hrimg.depth() != CV_64F)
		hrimg.convertTo(hrimg, CV_64F);
	int htrim = hrimg.rows - hrimg.rows % s;
	int wtrim = hrimg.cols - hrimg.cols % s; 

	Mat imtrim = hrimg(Rect(0,0,wtrim,htrim));

	int h_lr = htrim / s;
	int w_lr = wtrim / s;

	if (s % 2 == 1)
		cout << "Have not being done!" << endl;
	else if (s % 2 == 0)
	{
		//int sampleshift = s / 2;
		int kernelsize = ceil(sigma * 3) * 2 + 2;
		Mat blurimg;

		myGaussianBlur(imtrim, blurimg, 6);

		//Mat out_print = blurimg(Rect(0, 0, 10, 10));
		//cout << cv::format(out_print, Formatter::FMT_NUMPY) << endl;

		lrimg = Mat::zeros(h_lr, w_lr, CV_64FC1);

		for (int rl = 0; rl < h_lr; rl++)
		{
			int r_hr_sample = (rl)*s ;
			for (int cl = 0; cl < w_lr; cl++)
			{
				int c_hr_sample = (cl)*s;
				lrimg.at<double>(rl, cl) = blurimg.at<double>(r_hr_sample, c_hr_sample);
			}
		}
	}
	
}

void UpsampleAndBlur(Mat diff_lr, Mat& diff_hr, int zooming, double Gau_sigma)
{
	int h_hr = diff_lr.rows * zooming;
	int w_hr = diff_lr.cols * zooming;

	Mat upsampled = Mat::zeros(h_hr, w_hr, CV_64FC1);
	if (zooming == 2)
	{
		for (int rl = 0; rl < diff_lr.rows; rl++)
		{
			int rh = (rl) * zooming;
			for (int c1 = 0; c1 < diff_lr.cols; c1++)
			{
				int ch = (c1) * zooming;
				upsampled.at<double>(rh, ch) = diff_lr.at<double>(rl, c1);
			}
		}
		int kernelsize = ceil(Gau_sigma * 3) * 2 + 2;
		myGaussianBlur(upsampled, diff_hr, 6);
	}
	else
	{
		cout << "zoom not suppose!!!!!!!!!!!!!" << endl;
	}
	//Mat out_print = diff_hr(Rect(0, 0, 10, 10));
	//cout << cv::format(out_print, Formatter::FMT_NUMPY) << endl;
}

void ComputeGrad(Mat Img, std::vector<Mat> &grad)
{
	if (Img.channels() != 1)
		cvtColor(Img, Img, COLOR_RGB2GRAY);
	Img.convertTo(Img, CV_64F);

	double k1[1][3] = { -0.5,0,0.5 };  //水平方向的核
	double k2[3][1] = { -0.5,0,0.5 };  //垂直的核
	Mat gradx, grady;

	Mat Kore = Mat(1, 3, CV_64FC1, k1);
	Mat Kore2 = Mat(3, 1, CV_64FC1, k2);

	filter2D(Img, gradx, -1, Kore, Point(-1, -1), 0, BORDER_REFLECT);
	filter2D(Img, grady, -1, Kore2, Point(-1, -1), 0, BORDER_REFLECT);

	grad.push_back(gradx);
	grad.push_back(grady);
}

cv::Mat generalImgFromGpp(Mat img_y, std::vector<dlib::matrix<double>> grad_high, int zooming, double Gau_sigma, double beta)
{
	Mat gradtmp1, gradtmp2;
	std::vector<Mat> GradAvg;
	converMatrix2Mat(grad_high[0], gradtmp1);
	GradAvg.push_back(gradtmp1);
	converMatrix2Mat(grad_high[1], gradtmp2);
	GradAvg.push_back(gradtmp2);

	Mat div_exp;
	ComputeDivergence(GradAvg, div_exp);

	Mat img;
	cv::resize(img_y, img, Size(img_y.size()*zooming), 0, 0, INTER_LINEAR);

	double tau = 0.2;

	for (int iter = 0; iter < 30; iter++)
	{
		Mat img_low;
		GenerateLRImage_GaussianKernel(img, img_low, zooming, Gau_sigma);
		Mat diff = img_low - img_y;
		Mat GradDir1;
		UpsampleAndBlur(diff, GradDir1, zooming, Gau_sigma);

		std::vector<Mat> grad_img;
		ComputeGrad(img, grad_img);
		Mat div_img;
		ComputeDivergence(grad_img, div_img);
		Mat GradDir2 = div_img - div_exp;
		Mat GradDir = GradDir1 - beta * GradDir2;

		img = img - tau * GradDir;
	}
	return img;
}

