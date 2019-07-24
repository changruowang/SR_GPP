#pragma once
#include <dlib/matrix.h>
#include <opencv2/opencv.hpp>


using namespace dlib;
using namespace cv;

std::vector<matrix<double>> grad_gpp(std::vector<Mat> grad, int zooming);


