#include "Header.h"

double angle(int x1, int y1, int x2, int y2, int x3, int y3)	//餘弦值
{
	double dx1 = x1 - x3;
	double dy1 = y1 - y3;
	double dx2 = x2 - x3;
	double dy2 = y2 - y3;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

int dist(int x1, int y1, int x2, int y2)
{
	double tmp;
	tmp = sqrt(pow(int(x1 - x2), 2) + pow(int(y1 - y2), 2));
	return (int)tmp;
}

int dist(Point a, Point b)
{
	double tmp;
	tmp = sqrt(pow(int(a.x - b.x), 2) + pow(int(a.y - b.y), 2));
	return (int)tmp;
}

int main()
{
	nLoop = 5;
	bDebug = TRUE;
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
	//cap.set(CAP_PROP_CONTRAST, 20);
	//cap.set(CAP_PROP_BRIGHTNESS, 10);
	//cap.set(CAP_PROP_GAMMA, 200);

	cap.set(CAP_PROP_EXPOSURE, -6);			//曝光，此camera原始值-5
	cap.set(CAP_PROP_CONTRAST, 20);			//對比，此camera原始值20
	cap.set(CAP_PROP_BRIGHTNESS, 200);		//亮度，此camera原始值10
	cap.set(CAP_PROP_GAMMA, 100);		//Gamma，此camera原始值200

	char testtemp[100];
	cv::String text;
	IplImage iplimage;
	Mat match;

	//for (int i = 0; i < 24; i++)
	//{
	//	sprintf(testtemp, "dice%02d.jpg", i + 1);
	//	diceImage[i] = imread(testtemp, 1);
	//}
	dicebase[0] = imread("dicebase1.jpg", 1);
	dicebase[1] = imread("dicebase2.jpg", 1);

	while (1)
	{
		totlecircles.clear();

		if (bDebug)
		{
			nLastTime = GetTickCount();
			for (int i = 0; i < nLoop; i++)
			{
				cap >> frame;
				checkDicePoint(frame);

				if (i == nLoop - 1)
				{
					nNowTime = GetTickCount();
					nShowTime = nNowTime - nLastTime;
					showCircles = totlecircles;
					showLinep = totlelinesP;
					points.clear();
					for (int k = 0; k < showCircles.size(); k++)
					{
						points.push_back(Point(showCircles[k][0], showCircles[k][1]));
					}
				}
				frame.copyTo(frame1);

				//先用圖片比對確認三顆骰子的位置
				checkDicePoint(points);
				//用HoughCircles尋找骰子點數圓心
				drawCircle(frame1, showCircles);
				//將找出來圓心進行相對位置判斷，將應該存在但未找出的圓心進行補點動作，計算出各種骰子各有幾顆
				matchDices(CV_TM_CCOEFF_NORMED);

				//frame_watershed.zeros(frame.size(), 1);
				//watershedMat(frame, frame_watershed);
				//imshow("frame_watershed", frame_watershed);

				sprintf(testtemp, "Total number of dice points: %d", (int)showCircles.size());
				text = testtemp;
				putText(frame1, text, Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				sprintf(testtemp, "check use %d msec", nShowTime);
				text = testtemp;
				putText(frame1, text, Point(10, 65), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				imshow("frame", frame);
				//imshow("framegray", framegray);
				imshow("frame1", frame1);
				

				//imshow("contours", contours);

				switch (waitKey(1))
				{
				case 0x1B://VK_ESCAPE:
					return;
					break;
				case 0x0D://VK_RETURN:
					//imwrite("dices.jpg", frame);
					//matchImage(CV_TM_SQDIFF_NORMED);
					//matchImage(CV_TM_CCORR_NORMED);
					//matchImage(CV_TM_CCOEFF_NORMED);
					diceType.clear();
					for (int i = 0; i < showCircles.size(); i++) diceType.push_back(0);
					drawDice1(diceType, showCircles);
					break;
				case 'a':
				case 'A':
				{
				}
				break;
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
					checkDicePoint(frame);
				}
				frame.copyTo(frame1);
				drawCircle(frame1, totlecircles);
				nNowTime = GetTickCount();

				sprintf(testtemp, "Total number of dice points: %d", (int)totlecircles.size());
				text = testtemp;
				putText(frame1, text, Point(20, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				sprintf(testtemp, "check use %d msec", nNowTime - nLastTime);
				text = testtemp;
				putText(frame1, text, Point(20, 65), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

				imshow("frame1", frame1);

				//matchImage(CV_TM_CCOEFF);
				break;
			default:
				break;
			}
		}

	}
}

void calcCircles(const Mat &input, vector<Vec3f> &circles)
{
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
	Mat contours;
	Canny(input, contours, 100, 100);
	//vector<Vec4i> hierarchy;
	//pContours.clear();
	//findContours(contours, pContours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//for (int i = 0; i < pContours.size(); i++) {
	//	drawContours(frame1, pContours, i, Scalar(255, 0, 255), 1, 8, hierarchy);
	//}
	circles.clear();
	HoughCircles(contours, circles, CV_HOUGH_GRADIENT, 1, 12, 60, 10, 4, 6);
	
}

void calcCircles_New(const Mat &input, vector<Vec3f> &circles)
{
	IplImage *frame, *change, *getHSV;
	frame = &IplImage(input);
	CvSize ImageSize = cvSize(frame->width, frame->height);
	change = cvCreateImage(ImageSize, IPL_DEPTH_8U, 1);
	getHSV = cvCreateImage(ImageSize, IPL_DEPTH_8U, 3);

	cvSmooth(frame, getHSV, CV_BILATERAL, 7, 0, 600, 600);
	cvCvtColor(getHSV, change, CV_RGB2GRAY);

	//cvShowImage("BILATERAL", getHSV);

	cvSmooth(change, change, CV_MEDIAN, 3, 0, 0, 0);
	cvSmooth(change, change, CV_GAUSSIAN, 3, 3, 1, 1);

	cvNot(change, change);

	cvThreshold(change, change, 50, 255, CV_THRESH_BINARY);

	cvCanny(change, change, 15, 255, 3);

	cvDilate(change, change, 0, 1);
	cvErode(change, change, 0, 1);

	//cvShowImage("change", change);
	
	Mat contours;
	contours = cvarrToMat(change);
	circles.clear();
	HoughCircles(contours, circles, CV_HOUGH_GRADIENT, 1, 10, 60, 12, 1, 8);

}

void drawCircle(Mat &input, const vector<Vec3f> &circles)
{
	for (int i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//int radius = cvRound(circles[i][2]);
		circle(input, center, 7, Scalar(255, 0, 255), 2, 8, 0);
	}
}


void checkDicePoint(Mat image)
{
	vector<Vec3f> circles, circles2;

	//cvtColor(image, framegray, CV_BGR2GRAY);
	//GaussianBlur(framegray, framegray, Size(1, 1), 0, 0, BORDER_CONSTANT);			//高斯平滑
	//threshold(framegray, framegray, 55, 255, THRESH_BINARY);
	//adaptiveThreshold(framegray, framegray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);

	int nSize = (int)totlecircles.size();
	//calcCircles(framegray, circles);
	calcCircles_New(image, circles);

	for (int i = 0; i < circles.size(); i++)
	{		
		BOOL bCheck = TRUE;

		//去除不在圖片比對出來的骰子範圍內的圓心
		for (int m = 0; m < 3; m++)
		{
			if (checkPointInDice(circles[i][0], circles[i][1]) == 0)
			{
				bCheck = FALSE;
				break;
			}
		}

		//去除跟其它圓心距離太近的圓心
		if (bCheck)
		{
			for (int k = 0; k < totlecircles.size(); k++)
			{
				//double len = sqrt(pow(int(totlecircles[k][0] - circles[i][0]), 2) + pow(int(totlecircles[k][1] - circles[i][1]), 2));
				int len = dist(totlecircles[k][0], totlecircles[k][1], circles[i][0], circles[i][1]);
				if (len < 10)
				{
					bCheck = FALSE;
					break;
				}
			}
		}

		if (bCheck)
		{
			totlecircles.push_back(circles[i]);
		}
	}
}

int checkPointInDice(int x, int y)
{
	for (int i = 0; i < 3; i++)
	{
		if (x >= diceRect[i].x && x <= diceRect[i].x + diceRect[i].width &&
			y >= diceRect[i].y && y <= diceRect[i].y + diceRect[i].height)
		{
			return i + 1;
		}
	}
	return 0;
}

int checkPointInDice(Point point)
{
	for (int i = 0; i < 3; i++)
	{
		if (point.x >= diceRect[i].x && point.x <= diceRect[i].x + diceRect[i].width &&
			point.y >= diceRect[i].y && point.y <= diceRect[i].y + diceRect[i].height)
		{
			return i + 1;
		}
	}
	return 0;
}

void matchDices(int method)
{
	Mat temp;
	frame.copyTo(temp);

	double val = 0.0;
	Point loc;

	int num = 0;
	for (int check = 0; check < 3; check++)
	{
		for (int i = 0; i < 2; i++)
		{
			Point tempLoc;
			double tempVal;

			tempVal = grayMatch(temp, dicebase[i], tempLoc, method);
			if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
			{
				if (tempVal < val || i == 0)
				{
					num = i;
					loc = tempLoc;
					val = tempVal;
				}
			}
			else
			{
				if (tempVal > val || i == 0)
				{
					num = i;
					loc = tempLoc;
					val = tempVal;
				}
			}
		}

		Rect rect = Rect(0, 0, 0, 0);
		if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
		{
			if (method == CV_TM_SQDIFF_NORMED)
			{
				if (val < 0.6)
				{
					rect = Rect(loc, Point(loc.x + dicebase[num].cols, loc.y + dicebase[num].rows));
				}
			}
			else
			{
				rect = Rect(loc, Point(loc.x + dicebase[num].cols, loc.y + dicebase[num].rows));
			}
		}
		else
		{
			if (method == CV_TM_CCORR_NORMED || method == CV_TM_CCOEFF_NORMED)
			{
				if (val > 0.4)
				{
					rect = Rect(loc, Point(loc.x + dicebase[num].cols, loc.y + dicebase[num].rows));
				}
			}
			else
			{
				rect = Rect(loc, Point(loc.x + dicebase[num].cols, loc.y + dicebase[num].rows));
			}
		}
		diceRect[check] = rect;
		rectangle(frame1, rect, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(temp, rect, Scalar(255, 255, 255), CV_FILLED);		
	}
}

void matchImage(int method)
{
	nLastTime = GetTickCount();
	Mat temp, matchshow;
	double val = 0.0;
	Point loc;
	int nTotal = 0;

	char testtemp[100];
	cv::String text;

	frame.copyTo(temp);
	frame.copyTo(matchshow);

	int num = 0;
	for (int check = 0; check < 3; check++)
	{
		for (int i = 0; i < 24; i++)
		{
			//重覆圖片跳過，縮短時間
			if (i == 2 || i == 3 || i == 14 || i == 15 || i == 18 || i == 19) continue;

			Point tempLoc;
			double tempVal;

			tempVal = grayMatch(temp, diceImage[i], tempLoc, method);
			///////////////////////////////////////////////////////////////////////////////////////////////////////////
			/*
			int result_cols = image.cols - tepl.cols + 1;
			int result_rows = image.rows - tepl.rows + 1;

			Mat result = Mat(result_cols, result_rows, CV_32FC1);
			matchTemplate(image, tepl, result, method);
			//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

			Point minLoc, maxLoc;
			double minVal, maxVal;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
			*/
			///////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
			{
				if (tempVal < val || i == 0)
				{
					num = i;
					loc = tempLoc;
					val = tempVal;
				}
			}
			else
			{
				if (tempVal > val || i == 0)
				{
					num = i;
					loc = tempLoc;
					val = tempVal;
				}
			}
		}

		Rect rect = Rect(loc, Point(loc.x + diceImage[num].cols, loc.y + diceImage[num].rows));
		rectangle(matchshow, rect, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(temp, rect, Scalar(255, 255, 255), CV_FILLED);
		int nPoint = int(num / 4 + 1);
		sprintf(testtemp, "dice: %d(%d)", nPoint, num + 1);
		text = testtemp;
		putText(matchshow, text, Point(loc.x + diceImage[num].cols, loc.y), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
		nTotal += nPoint;
	}
	sprintf(testtemp, "Total number of dice points: %d", nTotal);
	text = testtemp;
	putText(matchshow, text, Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

	sprintf(testtemp, "check use %d msec", GetTickCount() - nLastTime);
	text = testtemp;
	putText(matchshow, text, Point(10, 65), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

	imshow("matchshow", matchshow);
	//imshow("temp", temp);
}

double basicMatch(Mat image, Mat tepl, Point &point, int method)
{
	//參數為CV_TM_SQDIFF、CV_TM_SQDIFF_NORMED時，計算結果較小時相似度較高，參數為CV_TM_CCORR、CV_TM_CCORR_NORMED、CV_TM_CCOEFF、CV_TM_CCOEFF_NORMED時，計算結果較大時相似度較高。
	int result_cols = image.cols - tepl.cols + 1;
	int result_rows = image.rows - tepl.rows + 1;

	Mat result = Mat(result_cols, result_rows, CV_32FC1);
	matchTemplate(image, tepl, result, method);
	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	Point minLoc, maxLoc;
	double minVal, maxVal;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	switch (method)
	{
	case CV_TM_SQDIFF:
	case CV_TM_SQDIFF_NORMED:
		point = minLoc;
		return minVal;
		break;
	case CV_TM_CCORR:
	case CV_TM_CCOEFF:
	case CV_TM_CCORR_NORMED:
	case CV_TM_CCOEFF_NORMED:
	default:
		point = maxLoc;
		return maxVal;
		break;
	}
}

double grayMatch(Mat image, Mat tepl, Point &point, int method)
{
	cvtColor(image, image, CV_BGR2GRAY);
	cvtColor(tepl, tepl, CV_BGR2GRAY);
	return basicMatch(image, tepl, point, method);
}

double binaryMatch(Mat image, Mat tepl, Point &point, int method)
{
	Mat src_img, dst_img;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	cvtColor(image, image, CV_BGR2GRAY);
	GaussianBlur(image, image, Size(1, 1), 0, 0, BORDER_CONSTANT);
	threshold(image, image, 55, 255, THRESH_BINARY);
	findContours(image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	src_img = Mat::zeros(image.size(), CV_8UC1);
	for (int i = 0;i < contours.size();i++)
	{
		drawContours(src_img, contours, i, Scalar(255), 1, 8, hierarchy);
	}

	cvtColor(tepl, tepl, CV_BGR2GRAY);
	GaussianBlur(tepl, tepl, Size(1, 1), 0, 0, BORDER_CONSTANT);
	threshold(tepl, tepl, 55, 255, THRESH_BINARY);
	findContours(tepl, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	dst_img = Mat::zeros(tepl.size(), CV_8UC1);
	for (int i = 0;i < contours.size();i++)
	{
		drawContours(dst_img, contours, i, Scalar(255), 1, 8, hierarchy);
	}

	return basicMatch(src_img, dst_img, point, method);
}

Mat RGBChange(Mat src)
{
	if (src.empty())
	{
		return src;
	}

	Mat imageRGB[3];
	split(src, imageRGB);

	for (int i = 0; i < 3; i++)
	{
		equalizeHist(imageRGB[i], imageRGB[i]);
	}

	merge(imageRGB, 3, src);

	return src;
}

Mat EnhanceChange(Mat src)
{
	if (src.empty())
	{
		return src;
	}

	Mat imageEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(src, imageEnhance, CV_8UC3, kernel);

	return imageEnhance;
}

Mat LogChange(Mat src)
{
	Mat imageLog(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = log(1 + src.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = log(1 + src.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = log(1 + src.at<Vec3b>(i, j)[2]);
		}
	}
	//歸一化到0~255
	normalize(imageLog, imageLog, 0, 255, CV_MINMAX);

	//轉換成8bit圖像顯示 
	convertScaleAbs(imageLog, imageLog);

	return imageLog;
}

Mat GammaChange(Mat src)
{
	Mat imageGamma(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			imageGamma.at<Vec3f>(i, j)[0] = (src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0])*(src.at<Vec3b>(i, j)[0]);
			imageGamma.at<Vec3f>(i, j)[1] = (src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1])*(src.at<Vec3b>(i, j)[1]);
			imageGamma.at<Vec3f>(i, j)[2] = (src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2])*(src.at<Vec3b>(i, j)[2]);
		}
	}

	//歸一化到0~255
	normalize(imageGamma, imageGamma, 0, 255, CV_MINMAX);

	//轉換成8bit圖像顯示
	convertScaleAbs(imageGamma, imageGamma);

	return imageGamma;
}

void watershedMat(Mat src_mat, Mat &dst_mat)
{
	// Create markers image
	Mat markers(src_mat.size(), CV_8U, Scalar(-1));

	//Rect(topleftcornerX, topleftcornerY, width, height);
	//top rectangle
	markers(Rect(0, 0, src_mat.cols, 5)) = Scalar::all(1);
	//bottom rectangle
	markers(Rect(0, src_mat.rows - 5, src_mat.cols, 5)) = Scalar::all(1);
	//left rectangle
	markers(Rect(0, 0, 5, src_mat.rows)) = Scalar::all(1);
	//right rectangle
	markers(Rect(src_mat.cols - 5, 0, 5, src_mat.rows)) = Scalar::all(1);
	//centre rectangle
	int centreW = src_mat.cols / 4;
	int centreH = src_mat.rows / 4;
	markers(Rect((src_mat.cols / 2) - (centreW / 2), (src_mat.rows / 2) - (centreH / 2), centreW, centreH)) = Scalar::all(2);
	markers.convertTo(markers, CV_BGR2GRAY);
	//imshow("markers", markers);

	//Create watershed segmentation object
	WatershedSegmenter segmenter;
	segmenter.setMarkers(markers);
	Mat wshedMask = segmenter.process(src_mat);
	Mat mask;
	convertScaleAbs(wshedMask, mask, 1, 0);
	double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
	bitwise_and(src_mat, src_mat, dst_mat, mask);
	dst_mat.convertTo(dst_mat, CV_8U);
}

void drawDice1(vector<int> &diceType, vector<Vec3f> circles)
{
	Mat checkDiceFrame;
	frame.copyTo(checkDiceFrame);

	char testtemp[100];
	cv::String text;
	int diceSize = 30;

	for (int i = 0; i < circles.size(); i++)
	{
		//設定骰子矩形範圍
		int x1, y1, x2, y2;
		x1 = circles[i][0] - diceSize;
		y1 = circles[i][1] - diceSize;
		x2 = x1 + diceSize * 2;
		y2 = y1 + diceSize * 2;

		//如果該點已經被標記為其它骰子，則不再判斷
		if (diceType[i] != 0) continue;

		BOOL bCheck = TRUE;
		for (int k = 0; k < circles.size(); k++)
		{
			if (i == k || diceType[k] != 0) continue;

			//如果有其它點在骰子範圍內，則另行判斷其它點的位置
			if (circles[k][0] >= x1 && circles[k][0] <= x2 && circles[k][1] >= y1 && circles[k][1] <= y2)
			{
				bCheck = FALSE;
				break;
			}
		}
		if (bCheck)
		{
			diceType[i] = 1;
			Rect rect = Rect(x1, y1, x2 - x1, y2 - y1);
			rectangle(checkDiceFrame, rect, Scalar(0, 0, 255), 2, 8, 0);
			putText(checkDiceFrame, "Dice 1", Point(x1, y1 - 20), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
		}
	}

	int num = 0;
	for (int i = 0; i < diceType.size(); i++)
	{
		if (diceType[i] == 1) num++;
	}
	sprintf(testtemp, "Dice 1: %d", num);
	text = testtemp;
	putText(checkDiceFrame, text, Point(10, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
	imshow("Dice 1", checkDiceFrame);
}

void drawDice2(vector<int> &diceType, vector<Vec3f> circles)
{

}

void drawDice3(vector<int> &diceType, vector<Vec3f> circles)
{

}

void drawDice4(vector<int> &diceType, vector<Vec3f> circles)
{

}

void drawDice5(vector<int> &diceType, vector<Vec3f> circles)
{

}

void drawDice6(vector<int> &diceType, vector<Vec3f> circles)
{

}

void checkDicePoint(vector<Point> points)
{
	//因為找圓問題，容許判斷相對位置時的距離誤差範圍
	int toleranceScope = 2;

	//所有點的總數
	int pointNum = points.size();

	//Debug訊息
	char testtemp[100];
	cv::String text;

	//用來記錄各點是否已被歸類為某種骰子的其中一點
	vector<int> diceType;
	//用來記錄各點是否已被歸類為某顆骰子的其中一點
	vector<int> diceNumber;
	//用來記錄是否為3連線的中間點
	vector<int> intermediate;
	for (int i = 0; i < pointNum; i++)
	{
		diceType.push_back(0);
		diceNumber.push_back(checkPointInDice(points[i]));
		intermediate.push_back(0);
	}

	//計算各點間的距離
	vector<vector<int>> vecDist;
	for (int i = 0; i < pointNum; i++)
	{
		vector<int> tempDist;
		for (int k = 0; k < pointNum; k++)
		{
			double dDist = dist(points[i], points[k]);
			tempDist.push_back(dDist);
		}
		vecDist.push_back(tempDist);
	}

	//以i為中間點判斷是否為3連線
	for (int i = 0; i < pointNum; i++)
	{
		for (int k = 0; k < pointNum; k++)
		{
			for (int s = 0; s < pointNum; s++)
			{
				if (i != k && i != s && k != s && angle(points[k].x, points[k].y, points[s].x, points[s].y, points[i].x, points[i].y) + 1 < 0.05)
				{
					if (vecDist[k][s] < 45//兩端距離太遠 非3連線
						&& fabs(vecDist[i][s] - vecDist[i][k]) <= toleranceScope) //兩端與中點距離在容許誤差範圍外，非3連線
					{
						if (diceType[i] == 3 && intermediate[i] == 1)
						{
							//如果中端已被歸類為某顆骰子的3連線，則更改該骰子為5連線
							for (int j = 0; j < pointNum; j++)
							{
								if (diceNumber[j] == diceNumber[i] && diceType[j] == 3)
								{
									diceType[j] = 5;
								}
							}
							diceType[i] = 5;
							diceType[k] = 5;
							diceType[s] = 5;
						}
						else
						{
							//如果中端未被歸類為某顆骰子的3連線，則設為新的3連線
							diceType[i] = 3;
							diceType[k] = 3;
							diceType[s] = 3;
							intermediate[i] = 1;
						}
					}
				}
			}
		}
	}

	//判斷兩個3連線是否為同一個6連線
	for (int i = 0; i < pointNum; i++)
	{
		for (int k = 0; k < pointNum; k++)
		{
			if (i != k && intermediate[i] == 1 && intermediate[k] == 1)
			{
				//以兩個3連線的中間點的距離來進行判斷，如果距離接近，則判斷為同一個6連線
				if (vecDist[i][k] < 25)
				{
					for (int j = 0; j < pointNum; j++)
					{
						if (diceNumber[j] == diceNumber[i] || diceNumber[j] == diceNumber[k])
						{
							diceType[j] = 6;
							diceNumber[j] = diceNumber[i];
						}
					}
					diceType[i] = 6;
					diceType[k] = 6;
					diceNumber[k] = diceNumber[i];
				}
			}
		}
	}

	//將屬於同一顆骰子的點連線起來作為debug標記
	for (int i = 0; i < pointNum; i++)
	{
		for (int k = 0; k < pointNum; k++)
		{
			if (i != k && diceNumber[i] == diceNumber[k] && diceNumber[i] != 0 && diceType[i] != 0 && diceType[k] != 0)
			{
				line(frame1, points[i], points[k], Scalar(255, 0, 0), 1);
			}
		}
	}

	int diceMsg[3] = { 0, 0, 0 };
	for (int i = 0; i < pointNum; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			if (diceNumber[i] == k + 1 && diceType[i] != 0) diceMsg[k]++;
		}
	}
	for (int i = 0; i < 3; i++)
	{
		sprintf(testtemp, "Dice %d : %d", i + 1, diceMsg[i]);
		text = testtemp;
		putText(frame1, text, Point(10, i * 25 + 90), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));
	}


	/*
	//判斷是否為點數1的骰子
	for (int i = 0; i < points.size(); i++)
	{
	for (int k = 0; k < points.size(); k++)
	{
	if (i != k)
	{
	//判斷標準1 : 是否與其它點在一定距離以上
	if (dist(points[i], points[k]) > 30)
	{
	continue;
	}

	//判斷標準2 : 與其它點在一定距離內時，判斷相對位置
	}
	}
	}
	*/
}
