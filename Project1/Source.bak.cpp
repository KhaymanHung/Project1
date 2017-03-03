#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <Windows.h>
#include <math.h>

using namespace cv;
using namespace std;

void DoVideoCapture();
void calcCircles(const Mat &input, vector<Vec3f> &circles);
void drawCircle(Mat &input, const vector<Vec3f> &circles);
void calcLines(const Mat &input, vector<Vec3f> &circles);
void drawLines(Mat &input, const vector<Vec4i> &lines);
void checkDicePoint();

vector<Vec3f> totlecircles, showCircles;
vector<Vec2f> lines;
int nLastTime = 0, nNowTime = 0, nShowTime = 0;
Mat frame, frame1, framegray, contours;
int nLoop = 20;
//int param = 30;
//Mat diceImage[7];
//Rect matchDiceRect[3];

int main()
{
	DoVideoCapture();
	return 0;
}

void DoVideoCapture()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
		return;

	//原始設定
	//cap.set(CAP_PROP_EXPOSURE, -5);
	cap.set(CAP_PROP_CONTRAST, 20);
	cap.set(CAP_PROP_BRIGHTNESS, 10);
	//cap.set(CAP_PROP_GAMMA, 200);

	cap.set(CAP_PROP_EXPOSURE, -7);			//曝光，此camera原始值-5
	//cap.set(CAP_PROP_CONTRAST, 20);			//對比，此camera原始值20
	//cap.set(CAP_PROP_BRIGHTNESS, 10);		//亮度，此camera原始值10
	cap.set(CAP_PROP_GAMMA, 100);		//Gamma，此camera原始值200

	char testtemp[100];
	cv::String text;

	//diceImage[0] = imread("dice1.jpg", 1);
	//diceImage[1] = imread("dice2.jpg", 1);
	//diceImage[2] = imread("dice3.jpg", 1);
	//diceImage[3] = imread("dice4.jpg", 1);
	//diceImage[4] = imread("dice5.jpg", 1);
	//diceImage[5] = imread("dice6.jpg", 1);
	//diceImage[6] = imread("POINT.jpg", 1);

	while (1)
	{
		//cap.set(CAP_PROP_EXPOSURE, param);			//曝光，此camera原始值-5
		//cap.set(CAP_PROP_CONTRAST, param);			//對比，此camera原始值20
		//cap.set(CAP_PROP_BRIGHTNESS, param);		//亮度，此camera原始值10
		//cap.set(CAP_PROP_GAMMA, param);		//Gamma，此camera原始值200

		totlecircles.clear();
		BOOL bDebug = TRUE;

		if (bDebug)
		{
			nLastTime = GetTickCount();
			for (int i = 0; i < nLoop; i++)
			{
				cap >> frame;
				checkDicePoint();

				if (i == nLoop - 1)
				{
					nNowTime = GetTickCount();
					nShowTime = nNowTime - nLastTime;
					showCircles = totlecircles;
				}

				///////////////////////////////////////////////////////////////////////////
				/*
				for (int dice = 0; dice < 6; dice++)
				{
					Mat result;
					result.create(frame.rows - diceImage[dice].rows + 1, frame.cols - diceImage[dice].cols + 1, CV_32FC1);

					matchTemplate(frame, diceImage[dice], result, CV_TM_SQDIFF);
					double minVal;
					Point minLoc;
					minMaxLoc(result, &minVal, 0, &minLoc, 0);

					if (minVal > 0)
					{
						rectangle(frame1, minLoc, Point(minLoc.x + diceImage[dice].cols, minLoc.y + diceImage[dice].rows), Scalar::all(0), 3);

						sprintf(testtemp, "dices image match: %d", dice);
						text = testtemp;
						putText(frame1, text, Point(20, 130 + dice * 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
					}
				}
				*/
				//Mat result, temp;
				//double minVal = 1.0;
				//Point minLoc;
				//int dice = 0;

				//frame.copyTo(temp);
				//for (int check = 0; check < 3; check++)
				//{
				//	result.create(temp.rows - diceImage[dice].rows + 1, temp.cols - diceImage[dice].cols + 1, CV_32FC1);

				//	matchTemplate(temp, diceImage[dice], result, CV_TM_CCOEFF);
				//	minMaxLoc(result, &minVal, 0, &minLoc, 0);

				//	if (minVal > 0)
				//	{
				//		rectangle(frame1, minLoc, Point(minLoc.x + diceImage[dice].cols, minLoc.y + diceImage[dice].rows), Scalar::all(0), 3);
				//		rectangle(temp, minLoc, Point(minLoc.x + diceImage[dice].cols, minLoc.y + diceImage[dice].rows), Scalar::all(0), CV_FILLED);
				//		//matchDiceRect[i] = Rect(minLoc, Point(minLoc.x + diceImage[dice].cols, minLoc.y + diceImage[dice].rows));
				//	}
				//	else
				//	{
				//		break;
				//	}
				//}// while (minVal > 0);
				///////////////////////////////////////////////////////////////////////////

				sprintf(testtemp, "Total number of dice points: %d", totlecircles.size());
				text = testtemp;
				putText(frame1, text, Point(20, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				//sprintf(testtemp, "param: %d", param);
				//text = testtemp;
				//putText(frame1, text, Point(20, 70), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				sprintf(testtemp, "check use %d msec", nShowTime);
				text = testtemp;
				putText(frame1, text, Point(20, 100), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				imshow("frame", frame);
				imshow("framegray", framegray);
				imshow("contours", contours);

				if (i == nLoop - 1)
				{
					imshow("frame1", frame1);
				}

				switch (waitKey(1))
				{
				case 0x1B://VK_ESCAPE:
					return;
					break;
				case 0x0D://VK_RETURN:
					imwrite("dices.jpg", frame);
					//imwrite("dices.jpg", framegray);
					break;
				//case '[':
				//	param--;
				//	break;
				//case ']':
				//	param++;
				//	break;
				default:
					break;
				}
			}
		}
		else
		{
			cap >> frame;
			imshow("frame", frame);

			switch (waitKey(1))
			{
			case 0x1B://VK_ESCAPE:
				return;
				break;
			case 0x0D://VK_RETURN:
				nLastTime = GetTickCount();
				for (int i = 0; i < nLoop; i++)
				{
					cap >> frame;
					checkDicePoint();
				}
				nNowTime = GetTickCount();

				sprintf(testtemp, "Total number of dice points: %d", totlecircles.size());
				text = testtemp;
				putText(frame1, text, Point(20, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				sprintf(testtemp, "check use %d msec", nNowTime - nLastTime);
				text = testtemp;
				putText(frame1, text, Point(20, 80), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				imshow("frame1", frame1);
				break;
			default:
				break;
			}
		}

	}
}

void calcCircles(const Mat &input, vector<Vec3f> &circles) {
	/*
	HoughCircles(	InputArray image, OutputArray circles, int method, double dp,
					double minDist, double param1 = 100, double param2 = 100,
					int minRadius = 0, int maxRadius = 0 );
	image: 輸入圖像 (灰度圖)
	circles: 存儲下面三個參數: x_{c}, y_{c}, r 集合的容器來表示每個檢測到的圓.
	method: 指定檢測方法. 現在OpenCV中只有霍夫梯度法(CV_HOUGH_GRADIENT)
	dp = 1: 累加器圖像的反比分辨率
	minDist: 檢測到圓心之間的最小距離
	param1 = 200: Canny邊緣函數的高閾值
	param2 = 100: 圓心檢測閾值.
	minRadius = 0: 能檢測到的最小圓半徑, 默認為0.
	maxRadius = 0: 能檢測到的最大圓半徑, 默認為0
	*/
	Canny(input, contours, 100, 100);
	HoughCircles(contours, circles, CV_HOUGH_GRADIENT, 1, 12, 60, 9, 4, 6);
	
}

void drawCircle(Mat &input, const vector<Vec3f> &circles) {
	for (int i = 0; i<circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(input, center, radius, Scalar(255, 0, 0), 2, 8, 0);
	}
}

void calcLines(const Mat &input, vector<Vec2f> &lines) {
	/*
	HoughLines( InputArray image, OutputArray lines, double rho, double theta, int threshold,
				double srn = 0, double stn = 0,	double min_theta = 0, double max_theta = CV_PI );
	image: 輸入圖像 (灰度圖)
	lines: 存儲參數, vector<Vec2f>.
	rho:
	theta:
	threshold:
	srn:
	stn:
	min_theta:
	max_theta:
	*/
	Canny(input, contours, 100, 100);
	HoughLines(contours, lines, 1, CV_PI / 180, 150, 0, 0);
}

void drawLines(Mat &input, const vector<Vec2f> &lines) {
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(input, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 255, 0), 1, CV_AA);
	}
}

void checkDicePoint()
{
	vector<Vec3f> circles, circles2;

	frame.copyTo(frame1);
	cvtColor(frame, framegray, CV_BGR2GRAY);
	//blur(framegray, framegray, Size(1, 1), Point(-1, -1));		//平均平滑
	GaussianBlur(framegray, framegray, Size(1, 1), 0, 0, BORDER_CONSTANT);			//高斯平滑
	//medianBlur(framegray, framegray, 7);							//中值濾波
	//normalize(framegray, framegray, 0, 255, NORM_MINMAX);
	threshold(framegray, framegray, 55, 255, THRESH_BINARY);

	int nSize = totlecircles.size();
	calcCircles(framegray, circles);
	for (int i = 0; i < circles.size(); i++)
	{
		if ((framegray.at<uchar>(circles[i][1], circles[i][0]) != 255))
		{
			continue;
		}

		//if ((framegray.at<uchar>(circles[i][1], circles[i][0]) != 255))
		//{
		//	continue;
		//}

		BOOL bCheck = TRUE;
		for (int k = 0; k < totlecircles.size(); k++)
		{
			double len = sqrt(pow(int(totlecircles[k][0] - circles[i][0]), 2) + pow(int(totlecircles[k][1] - circles[i][1]), 2));
			if (len < 10)
			{
				bCheck = FALSE;
				break;
			}
		}
		if (bCheck)
		{
			totlecircles.push_back(circles[i]);
		}
	}

	drawCircle(frame1, showCircles);

	//calcLines(framegray, lines);
	//drawLines(frame1, lines);
}
