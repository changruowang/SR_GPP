#include"run_gpp.h"

using namespace std;

int main(int argc, char *argv[])
{
	Mat LrImg, HrImg;

	if(argc < 2)
		return -1;	

	//HrImg = imread("/home/ruowang/workingfloder/SR_CPlus_LINUX/DateSet/HR1a.bmp", IMREAD_GRAYSCALE);
	LrImg = imread(argv[1], IMREAD_GRAYSCALE);
	//"/home/ruowang/workingfloder/SR_CPlus_LINUX/DateSet/LR1a.bmp"
	Mat Sr_img = run_gpp(LrImg, 2, 0.6, 0.05);
	imwrite("SR_IMG.bmp", Sr_img);
	
	imshow("SR", Sr_img);
	//imshow("HR", HrImg);
	cout << "end !" << endl;
	waitKey(20);
	
	return 0;
}


