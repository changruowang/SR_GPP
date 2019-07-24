#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <Dense> 
#include <Sparse>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sr_gpp.h"
#include "specialfunctions.h"

using namespace alglib;
using namespace dlib;
using namespace Eigen;
using namespace cv;
using namespace std;

typedef struct
{
	int r;
	int c;
	bool bFound;
}_pos_f;

typedef struct
{
	double Mag;
	double dist;
	double r;
	double c;
}_SearchResult;
typedef matrix<double, 0, 1> column_vector;
typedef matrix<double, 1, 0> row_vector;
typedef matrix<double> general_matrix;
typedef struct
{
	matrix<double, 1, 0> Values;
	matrix<int, 1, 0> IndexSet;

	int Number;
}_Coefw;

column_vector EdgePixelSigmaArray;
SparseMatrix <double> Coefw;
SparseMatrix <double> SumCoefw;


Mat read_csv(const char *filepath, Size img_size, int img_type)
{
	Mat image;
	image.create(img_size, img_type);
	string pixel;

	ifstream file(filepath, ifstream::in);
	if (!file)
		cout << "CSV read fail" << endl;

	int nl = image.rows;  // number of lines   
	int nc = image.cols; // number of columns   
	int eolElem = image.cols - 1;		//每行最后一个元素的下标
	int elemCount = 0;
	if (image.isContinuous())
	{
		nc = nc * nl;    // then no padded pixels   
		nl = 1;		  // it is now a 1D array   
	}
	for (int i = 0; i < nl; i++)
	{
		double* data = (double*)image.ptr<double>(i);
		for (int j = 0; j < nc; j++)
		{
			if (elemCount == eolElem) {
				getline(file, pixel, '\n');				//任意地读入，直到读到delim字符 '\n',delim字符不会被放入buffer中
				data[j] = (double)atof(pixel.c_str());	//将字符串str转换成一个双精度数值并返回结果
				elemCount = 0;							//计数器置零
			}
			else {
				getline(file, pixel, ',');				//任意地读入，直到读到delim字符 ','delim字符不会被放入buffer中
				data[j] = (double)atof(pixel.c_str());	//将字符串str转换成一个双精度数值并返回结果
				elemCount++;
			}
		}
	}
	return image;
}


void converCVMat2DlibMaxtrix(Mat& input, column_vector &result)
{
	if (input.cols != 1)   return;
	result.set_size(input.rows, 1);

	for (int r = 0; r < input.rows; r++)
		for (int c = 0; c < input.cols; c++)
			result(r, c) = input.at<double>(r, c);
}

void DlibM2EigenM(MatrixXd& em, matrix<double> dm)
{
	em.resize(dm.nr(), dm.nc());

	for (int r = 0; r < em.rows(); r++)
		for (int c = 0; c < em.cols(); c++)
			em(r, c) = dm(r, c);
}

matrix<double> myfun_grad(const column_vector x)
{
	double gamma = 5;
	matrix <double> FirstTerm = x - EdgePixelSigmaArray;

	MatrixXd e_x;
	DlibM2EigenM(e_x, x);
	MatrixXd e_SecondTerm = e_x.cwiseProduct(SumCoefw.cast<double>());
	MatrixXd e_ThirdTerm = Coefw * e_x;

	matrix<double, 0, 1> SecondTerm = dlib::mat(e_SecondTerm);
	matrix<double, 0, 1> ThirdTerm = dlib::mat(e_ThirdTerm);
	matrix<double, 0, 1> g = 2 * FirstTerm + 4 * gamma*SecondTerm - 4 * gamma*ThirdTerm;

	return g;
}


std::vector<_Coefw> g_NonZeroCoefw;
double myfun_f(const column_vector m)
{
	double gamma = 5;
	double SecondTerm = 0;
	int EdgePixelNumber = g_NonZeroCoefw.size();
	cout << "optimizer step++" << endl;
	for (int i = 0; i < EdgePixelNumber; i++)
	{
		matrix<double> Diff = m - m(i);
		matrix<double> Square = pointwise_multiply(Diff, Diff);

		//		if (((NonZeroCoefw[i].Values.nc()) != 0))
		if (((g_NonZeroCoefw[i].Values.nc()) != 0) && (max(g_NonZeroCoefw[i].Values == 0) != 1))
		{
			for (int j = 0; j < g_NonZeroCoefw[i].Values.nc(); j++)
				SecondTerm += g_NonZeroCoefw[i].Values(j) * Square(g_NonZeroCoefw[i].IndexSet(j));
		}
	}
	double f = dlib::sum(dlib::squared(m - EdgePixelSigmaArray)) + gamma * SecondTerm;

	return f;
}

Mat myIndexMat(Mat date, Mat index, int rowsort)
{
	CV_Assert(date.rows == index.rows);
	CV_Assert(date.cols == index.cols);
	CV_Assert(index.type() == 4 && date.type() == 5);
	Mat result = Mat::zeros(date.size(), date.type());

	for(int width=0; width < index.cols; width++)
		for (int height = 0; height < index.rows; height++)
		{    
			int pos = index.at<int>(height, width);
			if (rowsort)
				result.at<double>(height, width) = date.at<double>(height, pos);
			else 
				result.at<double>(height, width) = date.at<double>(pos, width);
		}
	return 	result;
}

void myIndexMatrix(row_vector m, matrix<int, 1, 0> index, row_vector& out)
{
	//如果要按列索引， 将输入矩阵转置
	out.set_size(index.nr(), index.nc());
	for(int r=0; r < index.nr(); r++)
		for (int c = 0; c < index.nc(); c++)
			out(r, c) = m(r, index(r, c));
}

SparseMatrix <double> creatSparseMaxtric(int* rows, int* cols, double* Values, int Rows, int Cols, int terms)
{
	SparseMatrix <double> A(Rows, Cols);

	std::vector <Triplet<double>> triplets;

	for (int i = 0; i < terms; ++i, ++rows,++cols, ++Values)
		if(abs(*Values) > 1e-8 )
			triplets.emplace_back(*rows, *cols, *Values);    // 填充Triplet

	A.setFromTriplets(triplets.begin(), triplets.end());    // 初始化系数矩阵

	return A;
}

SparseMatrix <double> creatSparseMaxtric(double* Values, int Rows, int Cols, int terms)
{
	SparseMatrix <double> A(Rows, Cols);

	std::vector <Triplet<double>> triplets;

	for (int i = 0; i < terms; ++i, ++Values)
		if (abs(*Values) > 1e-8)
			triplets.emplace_back(0, i, *Values);    // 填充Triplet

	A.setFromTriplets(triplets.begin(), triplets.end());    // 初始化系数矩阵

	return A;
}

SparseMatrix <double> creatSparseMaxtric(std::vector<int> rows, std::vector<int> cols, std::vector<double> Values, int Rows, int Cols, int terms)
{
	SparseMatrix <double> A(Rows, Cols);

	std::vector <Triplet<double>> triplets;

	for (int i = 0; i < terms; ++i)
		triplets.emplace_back(rows[i], cols[i], Values[i]);    // 填充Triplet
	
	A.setFromTriplets(triplets.begin(), triplets.end());    // 初始化系数矩阵

	return A;
}

void myMaxtrixSortIdx(row_vector m, matrix<int, 1, 0>& index, int flag)
{
	index.set_size(m.nr(), m.nc());
	int* index_one = new int[m.nc()];

	for (int r = 0; r < m.nr(); r++)
	{ 
		matrix<double> tmp = rowm(m, r);
		for (int i = 0; i < m.nc(); i++)
			index_one[i] = i;
		//排序
		for (int i = 0; i < m.nc() - 1; i++)
		{
			double x_tmp = tmp(i);
			for (int j = i + 1; j < m.nc(); j++)
			{
				double x = tmp(j);
				if (x < x_tmp)
				{
					int cg_tmp = index_one[j];
					index_one[j] = index_one[i];
					index_one[i] = cg_tmp;

					double value_tmp = tmp(j);
					tmp(j) = tmp(i);
					tmp(i) = value_tmp;

					x_tmp = x;
				}
			}
		}
		set_rowm(index, r) = mat(index_one, 1, m.nc());
	}
	delete index_one;
}

void mySparseToTriplet(SparseMatrix<double>& sm, Triplet<double>* result, int print = 1)
{
	if (print)
		std::cout << "#Rows:" << sm.rows()  << "  #Cols:" << sm.cols() << "  #Terms:" << sm.nonZeros() << endl;

	for (int k = 0; k < sm.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(sm, k); it; ++it)
		{
			if(print)
				std::cout << "(" << it.row() << ","<< it.col() << ")" << "  " << it.value() << endl;
			
			(*result++) = Triplet<double>(it.row(), it.col(), it.value());	
		}
	}

}

void mySparseMatrix_Sum(SparseMatrix<double> sm, SparseMatrix<double>& result)
{
	Triplet<double>* triplets = new Triplet<double>[sm.nonZeros()];
	Triplet<double>* pt = triplets;
	mySparseToTriplet(sm, triplets, 0);
	int cols = sm.cols();
	double* valueTmp = new double[cols];

	double* p = valueTmp;

	for (int i = 0; i < cols; i++)
		*p++ = 0;

	for (int i = 0; i < sm.nonZeros(); i++, pt++)
		*(valueTmp + pt->col()) += pt->value();

	result = creatSparseMaxtric(valueTmp, 1, cols, cols);

	delete valueTmp;
	delete triplets;
}


void saveMat(cv::Mat inputMat,const char* filename)
{
	FILE* fpt = fopen(filename, "w");
	int rows = inputMat.rows;
	int clos = inputMat.cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < clos; j++)
		{
			if (j < clos - 1)
				fprintf(fpt, "%.6f,", inputMat.at<double>(i, j));
			else
				fprintf(fpt, "%.6f\n", inputMat.at<double>(i, j));
		}
	}
	fclose(fpt);

}


double MapSigmaValue(double *a, double *b, double Sigma_low, int SeqLength)
{
	bool bFound = false;
	double Sigma_high;
	for (int i = 0; i < SeqLength - 1; i++)
	{
		if ((a[i] <= Sigma_low) && (a[i + 1] > Sigma_low))
		{
			Sigma_high = b[i] + (b[i + 1] - b[i])*(Sigma_low - a[i]) / 0.2;
			bFound = true;
			break;
		}
	}
	if ((bFound == false) && Sigma_low >= a[SeqLength - 1])
		Sigma_high = Sigma_low;
	else if (bFound == false)
	{
		std::cout << "cant map" << endl;
		string input;
		cin >> input;
		Sigma_high = 0;
	}
	return Sigma_high;
}


std::vector<general_matrix> grad_gpp(std::vector<Mat> grad, int zooming)
{
	double lambda_low = 1.6;
	double lambda_high;
	//if (zooming == 2)
	//{
	lambda_high = 1.63;
	double a[] = { 0, 0.100000, 0.300000, 0.500000, 0.700000, 0.900000,
		1.100000, 1.300000, 1.500000, 1.700000, 1.900000, 2.100000, 2.300000, 
		2.500000, 2.700000, 2.900000, 3.100000, 3.300000, 3.500000, 3.700000, 
		3.900000, 4.100000, 4.300000, 4.500000, 4.700000, 4.900000 };
	double b[] = { 0, 0.1, 0.534504, 0.650185, 0.724020, 0.792875,
		0.872219, 0.973969, 1.104349, 1.262270, 1.437783, 1.624416, 1.815510,
		2.020180, 2.210189, 2.404601, 2.633510, 2.844732, 3.071728, 3.296107,
		3.525818, 3.794026, 4.012113, 4.262773, 4.404142, 4.682555 };

	//}
	Mat GradMagnitude, tmp1,tmp2;

	GradMagnitude.create(grad[0].size(), CV_64FC1);
	cv::pow(grad[0], 2, tmp1);
	cv::pow(grad[1], 2, tmp2);
	cv::sqrt(tmp1 + tmp2, GradMagnitude);

	//Mat out_print1 = grad[0](Rect(0, 0, 10, 10));
	//std::cout << cv::format(out_print1, Formatter::FMT_MATLAB) << endl;

	matrix<int> EdgePixelMap;
	Mat SigmaMap;
	EdgePixelMap = dlib::zeros_matrix<int>(GradMagnitude.rows, GradMagnitude.cols);
	SigmaMap = Mat::zeros(GradMagnitude.size(), CV_64FC1);

	for (int r = 1; r < GradMagnitude.rows-1; r++)
	{
		std::cout << r << endl;
		for (int c = 1; c < GradMagnitude.cols-1; c++)
		{
			double gradc = grad[0].at<double>(r, c);
			double gradr = grad[1].at<double>(r, c);
			double absgradc, PosNeighborMag;
			double absgradr, NegNeighborMag;
			double dcSign, drSign, dr1, dc1, w1, w2, dr2, dc2;

			if (GradMagnitude.at<double>(r, c) > 0.000001)
			{
				if(gradc > 0)
					dcSign = 1;
				else if(gradc < 0)
					dcSign = -1;
				else
					dcSign = 0;      
					
				if(gradr > 0)
					drSign = 1;
				else if(gradr < 0)
					drSign = -1;
				else
					drSign = 0;      
				absgradr = abs(gradr);
				absgradc = abs(gradc);
				if (absgradc == 0 && absgradr > 0)
				{
					dr1 = 1;
					dc1 = 0;
					w1 = 1;
					dr2 = 1;
					dc2 = 0;
				}
				else if (absgradr > absgradc)
				{
					dr1 = 1;
					dc1 = 0;
					w1 = 1 - absgradc / absgradr;
					dr2 = 1;
					dc2 = 1;
				}
				else if(absgradr == absgradc)
				{
					dr1 = 1;
					dc1 = 1;
					w1 = 1;
					dr2 = 1;
					dc2 = 1;
				}
				else if (absgradc > absgradr)
				{
					dr1 = 0;
					dc1 = 1;
					w1 = 1 - absgradr / absgradc;
					dr2 = 1;
					dc2 = 1;
				}
				else if (absgradc > 0 && absgradr == 0)
				{
					dr1 = 0;
					dc1 = 1;
					w1 = 1;
					dr2 = 0;
					dc2 = 1;
				}
				else
					std::cout <<"should not have such a case" << endl;
				PosNeighborMag = w1 * GradMagnitude.at<double>(r + drSign * dr1, c + dcSign * dc1) + (1 - w1)*GradMagnitude.at<double>(r + drSign * dr2, c + dcSign * dc2);
				NegNeighborMag = w1 * GradMagnitude.at<double>(r - drSign * dr1, c - dcSign * dc1) + (1 - w1)*GradMagnitude.at<double>(r - drSign * dr2, c - dcSign * dc2);
			
				if (GradMagnitude.at<double>(r, c) > PosNeighborMag && GradMagnitude.at<double>(r, c) > NegNeighborMag)
				{
					_SearchResult ProfilePoints[60];
					std::vector<_SearchResult> SearchResultPos;
					std::vector<_SearchResult> SearchResultNeg;
					EdgePixelMap(r, c) = 1;
					for (int SearchDirectionIdx = 0; SearchDirectionIdx <2; SearchDirectionIdx++)
					{
						int Sign;
						if(SearchDirectionIdx == 1)
							Sign = 1;
						else
							Sign = -1;  

						row_vector Shift1_R, Shift1_C, dist1, Shift2_R, Shift2_C, dist2;
						if (gradr != 0)
						{
							Shift1_R = dlib::linspace(1,30,30);
							Shift1_C = Shift1_R * gradc / gradr;
							dist1 = dlib::sqrt(dlib::squared(Shift1_R) + dlib::squared(Shift1_C));
						}
						if (gradc != 0)
						{
							Shift2_C = dlib::linspace(1, 30, 30);
							Shift2_R = Shift2_C * gradr / gradc;
							dist2 = dlib::sqrt(dlib::squared(Shift2_R) + dlib::squared(Shift2_C));
						}
						row_vector Shift_R = dlib::join_rows(Shift1_R, Shift2_R);
						row_vector Shift_C = dlib::join_rows(Shift1_C, Shift2_C);

						row_vector MainCoor = join_rows(linspace(1, 1, 30), 2 * linspace(1, 1, 30));
						row_vector distboth = join_rows(dist1, dist2);
						row_vector SearchR, SearchC, SearchCoor;
						matrix<int, 1, 0> index;

						myMaxtrixSortIdx(distboth, index, 1);
						myIndexMatrix(Shift_R, index, SearchR);
						myIndexMatrix(Shift_C, index, SearchC);
						myIndexMatrix(MainCoor, index, SearchCoor);


						_SearchResult NextPoint = {
							GradMagnitude.at<double>(r,c),
							0,
							r,
							c
						};
						std::vector<_SearchResult> SearchResult;
						SearchResult.push_back(NextPoint);
						NextPoint = {
							0,
							0,
							0,
							0,
						};
						for (int i = 0; i < 30; i++)
						{
							int LastIdx = SearchResult.size();
							_SearchResult CurrentPoint = SearchResult[LastIdx-1];
							NextPoint.r = r + Sign * drSign*SearchR(i);
							NextPoint.c = c + Sign * drSign*SearchC(i);

							if (!((NextPoint.r >= 0) && (NextPoint.r <= (GradMagnitude.rows-1)) && (NextPoint.c >= 0) && (NextPoint.c <= (GradMagnitude.cols-1))))
								break;
							if (SearchCoor(i) == 1)
							{
								int c1 = floor(NextPoint.c);
								int c2 = ceil(NextPoint.c);
								if (c1 == c2)
									NextPoint.Mag = GradMagnitude.at<double>(NextPoint.r, NextPoint.c);
								else
								{
									double w1 = c2 - NextPoint.c;
									double w2 = NextPoint.c - c1;
									NextPoint.Mag = w1 * GradMagnitude.at<double>(NextPoint.r, c1) + w2 * GradMagnitude.at<double>(NextPoint.r, c2);
								}
									
							}
							else
							{
								int r1 = floor(NextPoint.r);
								int r2 = ceil(NextPoint.r);
								if(r1 == r2)
									NextPoint.Mag = GradMagnitude.at<double>(NextPoint.r, NextPoint.c);
								else
								{
									double w1 = r2 - NextPoint.r;
									double w2 = NextPoint.r - r1;
									NextPoint.Mag = w1 * GradMagnitude.at<double>(r1, NextPoint.c) + w2 * GradMagnitude.at<double>(r2, NextPoint.c);
								}
									
							}
							if(NextPoint.Mag >= CurrentPoint.Mag)
								break;
							else
							{
								NextPoint.dist = sqrt((NextPoint.r - r)*(NextPoint.r - r) + (NextPoint.c - c)*(NextPoint.c - c));
								SearchResult.push_back(NextPoint);
							}		
						}
						if (SearchDirectionIdx == 1)
							SearchResultPos.swap(SearchResult);
						else
							SearchResultNeg.swap(SearchResult);
					}
					int PosLength = SearchResultPos.size();
					int NegLength = SearchResultNeg.size();
					int ProfileLength = PosLength + NegLength - 1;
					for (int j = 0; j < ProfileLength; j++)
						ProfilePoints[j] = {
							GradMagnitude.at<double>(r, c),
							0,
							(double)r,
							(double)c
						};
					int idx = 0;
					for (int j = NegLength-1; j >= 1; j--)
					{
						ProfilePoints[idx] = SearchResultNeg[j];
						idx = idx + 1;
					}
					for (int j = 0; j < PosLength; j++)
					{
						ProfilePoints[idx] = SearchResultPos[j];
						idx = idx + 1;
					}
					double Nominator = 0;
					double Denominator = 0;          
					
					for (int j = 0; j < ProfileLength; j++)
					{
						Nominator = Nominator + ProfilePoints[j].Mag * ProfilePoints[j].dist * ProfilePoints[j].dist;
						Denominator = Denominator + ProfilePoints[j].Mag;
					}
					double sigma;
					if (Denominator != 0)
						sigma = std::sqrt(Nominator / Denominator);
					else
					{
						printf("erro! Line 367");
						string erro;
						cin >> erro;
					}
						

					SigmaMap.at<double>(r, c) = sigma;
					
				}
			}
		}
	}

	int EdgePixelNumber = dlib::sum(EdgePixelMap);
	EdgePixelSigmaArray = dlib::zeros_matrix<double>(EdgePixelNumber, 1);
	matrix<int> EdgePixelIndex;
	EdgePixelIndex = dlib::zeros_matrix<int>(GradMagnitude.rows, GradMagnitude.cols);
	int idx = 0;

	for (int r = 0; r < GradMagnitude.rows; r++)
		for (int c = 0; c < GradMagnitude.cols; c++)
		{
			if (EdgePixelMap(r, c) == 1)
			{
				EdgePixelIndex(r, c) = idx;
				EdgePixelSigmaArray(idx) = SigmaMap.at<double>(r, c);
				idx = idx + 1;
			}
		}

	int* NonZeroIndexSet = new int[51200];
	double* NonZeroValues = new double[51200];

	std::vector<int> iData, jData;
	std::vector<double> wData;
	idx = 0;
	std::vector<_Coefw> NonZeroCoefw;

	for (int r = 0; r < GradMagnitude.rows; r++)
	{
		int NonZeroIndex = 0;

		for (int c = 0; c < GradMagnitude.cols; c++)
		{
			if (EdgePixelMap(r, c) == 1)
			{
				int i = EdgePixelIndex(r, c);
				for (int rs = -5; rs <= 5; rs++)
					for (int cs = -5; cs <= 5; cs++)
					{
						if ((!(rs == 0 && cs == 0)) && ((rs*rs + cs*cs) <= 25) && ((r + rs) >= 0) && ((r + rs) <= GradMagnitude.rows-1) && ((c + cs) >= 0) && ((c + cs) <= (GradMagnitude.cols-1)) && (EdgePixelMap(r + rs, c + cs) == 1))
						{
							int rj = r + rs;
							int cj = c + cs;
							int j = EdgePixelIndex(rj, cj);
							idx = idx + 1;
							iData.push_back(i);
							jData.push_back(j);

							double GradDiffSquare = cv::pow((grad[1].at<double>(r, c) - grad[1].at<double>(rj, cj)), 2) + cv::pow(grad[0].at<double>(r, c) - grad[0].at<double>(rj, cj) ,2);
							double ComputedCoefw = exp(-0.15 * GradDiffSquare - 0.08*(rs*rs + cs*cs));
							
							wData.push_back(ComputedCoefw);
							
							//cout << ">" << NonZeroIndex << endl;
							NonZeroIndexSet[NonZeroIndex] = j;
							NonZeroValues[NonZeroIndex] = ComputedCoefw;
							NonZeroIndex = NonZeroIndex + 1;
						}

					}
			}

		}

		_Coefw tmp;
		tmp.Number = NonZeroIndex;
		tmp.Values = dlib::mat(NonZeroValues, 1, NonZeroIndex);
		tmp.IndexSet = dlib::mat(NonZeroIndexSet,1,  NonZeroIndex);
		NonZeroCoefw.push_back(tmp);		
	
	}
	delete NonZeroIndexSet;
	delete NonZeroValues;

	Coefw = creatSparseMaxtric(iData, jData, wData, EdgePixelNumber, EdgePixelNumber, idx);
	mySparseMatrix_Sum(Coefw, SumCoefw);
	SumCoefw = SumCoefw.transpose();


	g_NonZeroCoefw.swap(NonZeroCoefw);
	column_vector NewSigmaArray = EdgePixelSigmaArray;

	//

	std::cout << "start optimizer" << endl;
	find_min(lbfgs_search_strategy(20),  // The 10 here is basically a measure of how much memory L-BFGS will use.
		objective_delta_stop_strategy(1e-5, 25).be_verbose(),  // Adding be_verbose() causes a message to be 
		myfun_f, myfun_grad, NewSigmaArray, -1);        // printed for each iteration of optimization.	
	std::cout << "end optimizer" << endl;


	general_matrix NewSigmaMap = dlib::zeros_matrix<double>(GradMagnitude.rows, GradMagnitude.cols);
	idx = 0;

	for (int r = 0; r < GradMagnitude.rows; r++)
		for (int c = 0; c < GradMagnitude.cols; c++)
		{
			if (EdgePixelMap(r, c) == 1)
			{
				NewSigmaMap(r, c) = NewSigmaArray(idx);
				idx = idx + 1;
			}
		}

	//cout << subm(NewSigmaMap, range(0, 10), range(0, 10)) << endl;

	std::vector<general_matrix> NewGradMatrix;  
	NewGradMatrix.push_back(dlib::zeros_matrix<double>(GradMagnitude.rows, GradMagnitude.cols));
	NewGradMatrix.push_back(dlib::zeros_matrix<double>(GradMagnitude.rows, GradMagnitude.cols));

	double AlphaLambda_high = sqrt(gammafunction(3 / lambda_high) / gammafunction(1 / lambda_high));
	double AlphaLambda_low = sqrt(gammafunction(3 / lambda_low) / gammafunction(1 / lambda_low));
	double ValueC_exceptSigma = lambda_high / lambda_low * AlphaLambda_high / AlphaLambda_low * gammafunction(1 / lambda_low) / gammafunction(1 / lambda_high);

	for(int r = 0; r < GradMagnitude.rows; r++)
		for (int c = 0; c < GradMagnitude.cols; c++)
		{
			bool bChangeSigma = false;
			double gradr;
			double gradc;
			_pos_f EdgePixel;

			int dcSign, drSign;
			if (EdgePixelMap(r, c) == 1)
			{
				EdgePixel.r = r;
				EdgePixel.c = c;
				bChangeSigma = true;
			}
			else
			{
				gradr = grad[1].at<double>(r, c);
				gradc = grad[0].at<double>(r, c);
				if (gradc > 0)
					dcSign = 1;
				else if (gradc < 0)
					dcSign = -1;
				else 
					dcSign = 0;

				if (gradr > 0)
					drSign = 1;
				else if (gradr < 0)
					drSign = -1;
				else
					drSign = 0;

				if (GradMagnitude.at<double>(r, c) > 0.000001)
				{
					_pos_f FoundEdgePixelPos, FoundEdgePixelNeg;
					for (int SearchDirectionIdx = 0; SearchDirectionIdx < 2; SearchDirectionIdx++)
					{
						row_vector dist1, dist2, Shift1_R, Shift1_C, Shift2_C, Shift2_R;
						row_vector SearchR, SearchC, SearchCoor;
						_pos_f FoundEdgePixel;
						int Sign;
						FoundEdgePixel.bFound = false;
						if(SearchDirectionIdx == 0 )
							Sign = 1;
						else
							Sign = -1;

						if (gradr != 0)
						{
							Shift1_R = dlib::linspace(1, 30, 30);
							Shift1_C = Shift1_R * gradc / gradr;
							dist1 = dlib::sqrt(dlib::squared(Shift1_R) + dlib::squared(Shift1_C));
						}
						if (gradc != 0)
						{
							Shift2_C = dlib::linspace(1, 30, 30);
							Shift2_R = Shift2_C * gradr / gradc;
							dist2 = dlib::sqrt(dlib::squared(Shift2_R) + dlib::squared(Shift2_C));
						}
						row_vector Shift_R = join_rows(Shift1_R, Shift2_R);
						row_vector Shift_C = join_rows(Shift1_C, Shift2_C);

						row_vector MainCoor = join_rows(linspace(1,1,30), 2*linspace(1, 1, 30));
						row_vector distboth = join_rows(dist1, dist2);

						matrix<int, 1, 0> index;
	
						myMaxtrixSortIdx(distboth, index, 1);
						myIndexMatrix(Shift_R, index, SearchR);
						myIndexMatrix(Shift_C, index, SearchC);
						myIndexMatrix(MainCoor, index, SearchCoor);

						_SearchResult NextPoint = {
							GradMagnitude.at<double>(r, c),
							0,
							r,
							c
						};
						std::vector<_SearchResult> SearchResult;
						SearchResult.push_back(NextPoint);
						NextPoint = {
							0,
							0,
							0,
							0
						};
						for (int i = 0; i < 30; i++)
						{
							int LastIdx = SearchResult.size();
							_SearchResult CurrentPoint = SearchResult[LastIdx-1];
							NextPoint.r = r + Sign * drSign*SearchR(i);
							NextPoint.c = c + Sign * dcSign*SearchC(i);
							if(!(NextPoint.r >= 0 && (NextPoint.r <= (GradMagnitude.rows-1)) && NextPoint.c >= 0 && (NextPoint.c <= (GradMagnitude.cols-1))))
								break;
							if (SearchCoor(i) == 1)
							{
								int c1 = floor(NextPoint.c);
								int c2 = ceil(NextPoint.c);
								if (c1 == c2)
									NextPoint.Mag = GradMagnitude.at<double>(NextPoint.r, NextPoint.c);
								else
								{
									double w1 = c2 - NextPoint.c;
									double w2 = NextPoint.c - c1;
									NextPoint.Mag = w1 * GradMagnitude.at<double>(NextPoint.r, c1) + w2 * GradMagnitude.at<double>(NextPoint.r, c2);
								}
								if (EdgePixelMap(NextPoint.r, c1) == 1)
								{
									FoundEdgePixel.r = NextPoint.r;
									FoundEdgePixel.c = c1;
									FoundEdgePixel.bFound = true;
								}
								if (EdgePixelMap(NextPoint.r, c2) == 1)
								{
									FoundEdgePixel.r = NextPoint.r;
									FoundEdgePixel.c = c2;
									FoundEdgePixel.bFound = true;
								}	
							}
							else
							{
								int r1 = floor(NextPoint.r);
								int r2 = ceil(NextPoint.r);

								if (r1 == r2)
									NextPoint.Mag = GradMagnitude.at<double>(NextPoint.r, NextPoint.c);
								else
								{
									double w1 = r2 - NextPoint.r;
									double w2 = NextPoint.r - r1;
									NextPoint.Mag = w1 * GradMagnitude.at<double>(r1, NextPoint.c) + w2 * GradMagnitude.at<double>(r2, NextPoint.c);
								}
								if (EdgePixelMap(r1, NextPoint.c) == 1)
								{
									FoundEdgePixel.r = r1;
									FoundEdgePixel.c = NextPoint.c;
									FoundEdgePixel.bFound = true;
								}
								if (EdgePixelMap(r2, NextPoint.c) == 1)
								{
									FoundEdgePixel.r = r2;
									FoundEdgePixel.c = NextPoint.c;
									FoundEdgePixel.bFound = true;
								}
							}
							if (NextPoint.Mag <= CurrentPoint.Mag)
								break;
							else
								SearchResult.push_back(NextPoint);
						}
						if(SearchDirectionIdx == 0)
							FoundEdgePixelPos = FoundEdgePixel;
						else
							FoundEdgePixelNeg = FoundEdgePixel;	
					}

					if (FoundEdgePixelPos.bFound || FoundEdgePixelNeg.bFound)
					{
						bChangeSigma = true;
						if (FoundEdgePixelPos.bFound && FoundEdgePixelNeg.bFound == false)
						{
							EdgePixel.r = FoundEdgePixelPos.r;
							EdgePixel.c = FoundEdgePixelPos.c;
						}
						else if (FoundEdgePixelPos.bFound == false && FoundEdgePixelNeg.bFound)
						{
							EdgePixel.r = FoundEdgePixelNeg.r;
							EdgePixel.c = FoundEdgePixelNeg.c;
						}
						else
						{
							double distPos = std::sqrt(std::pow((FoundEdgePixelPos.r - r),2) + std::pow((FoundEdgePixelPos.c - c),2));
							double distNeg = std::sqrt(std::pow((FoundEdgePixelNeg.r - r),2) + std::pow((FoundEdgePixelNeg.c - c),2));
							if (distPos <= distNeg)
							{
								EdgePixel.r = FoundEdgePixelPos.r;
								EdgePixel.c = FoundEdgePixelPos.c;
							}
							else
							{
								EdgePixel.r = FoundEdgePixelNeg.r;
								EdgePixel.c = FoundEdgePixelNeg.c;
							}		
						}
					}
				}
			}
			
			if (bChangeSigma)
			{
				bool b_infeasible = false;
				double Ratio;
				double Sigma_low = NewSigmaMap(EdgePixel.r, EdgePixel.c);
				double Sigma_high = MapSigmaValue(a, b, Sigma_low, sizeof(a)/sizeof(double));
				if (Sigma_high != 0)
				{
					double ValueC = ValueC_exceptSigma * Sigma_low / Sigma_high;
					double dist = std::sqrt(std::pow((r - EdgePixel.r), 2) + std::pow((c - EdgePixel.c), 2));
					Ratio = ValueC * std::exp(-std::pow((AlphaLambda_high*dist / Sigma_high),lambda_high) + std::pow((AlphaLambda_low*dist / Sigma_low), lambda_low));
					if ((abs(Ratio * grad[0].at<double>(r, c)) > 1) || (abs(Ratio * grad[1].at<double>(r, c)) > 1))
						b_infeasible = true;
				}	
				else
					b_infeasible = true;

				if (b_infeasible)
				{
					NewGradMatrix[0](r, c) = grad[0].at<double>(r, c);
					NewGradMatrix[1](r, c) = grad[1].at<double>(r, c);
				}
				else
				{
					NewGradMatrix[0](r, c) = Ratio * grad[0].at<double>(r, c);
					NewGradMatrix[1](r, c) = Ratio * grad[1].at<double>(r, c);
				}
			}
		}
	return NewGradMatrix;
}

