#pragma once
#include <dlib/matrix.h>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace dlib;
using namespace cv;

void converMatrix2Mat(matrix<double> m, Mat& out);
void ComputeDivergence(std::vector<Mat> grad, Mat& out);
void GenerateLRImage_GaussianKernel(Mat hrimg, Mat& lrimg, int s, double sigma);
void ComputeGrad(Mat Img, std::vector<Mat> &grad);
cv::Mat generalImgFromGpp(Mat img_y, std::vector<dlib::matrix<double>> grad_high, int zooming, double Gau_sigma, double beta);




