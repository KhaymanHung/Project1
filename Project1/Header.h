#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <Windows.h>
#include <math.h>

using namespace cv;
using namespace std;

void DoVideoCapture();
void calcCircles(const Mat &input, vector<Vec3f> &circles);
void calcCircles_New(const Mat &input, vector<Vec3f> &circles);
void drawCircle(Mat &input, const vector<Vec3f> &circles);

//餘弦值
double angle(int x1, int y1, int x2, int y2, int x3, int y3);

//兩點距離
int dist(int x1, int y1, int x2, int y2);
int dist(Point a, Point b);

void checkDicePoint(Mat image);
void matchDices(int method);
void matchImage(int method);
double basicMatch(Mat image, Mat tepl, Point &point, int method);
double grayMatch(Mat image, Mat tepl, Point &point, int method);
double binaryMatch(Mat image, Mat tepl, Point &point, int method);

/*
基於直方圖均衡化的圖像增強

直方圖均衡化是通過調整圖像的灰階分布，使得在0~255灰階上的分布更加均衡，提高了圖像的對比度，達到改善圖像主觀視覺效果的目的。對比度較低的圖像適合使用直方圖均衡化方法來增強圖像細節。
*/
Mat RGBChange(Mat src);

/*
基於拉普拉斯算子的圖像增強

使用中心為5的8鄰域拉普拉斯算子與圖像卷積可以達到銳化增強圖像的目的。
拉普拉斯算子可以增強局部的圖像對比度
*/
Mat EnhanceChange(Mat src);

/*
基於對數Log變換的圖像增強

對數變換可以將圖像的低灰度值部分擴展，顯示出低灰度部分更多的細節，將其高灰度值部分壓縮，減少高灰度值部分的細節，從而達到強調圖像低灰度部分的目的。
對於不同的底數，底數越大，對低灰度部分的擴展就越強，對高灰度部分的壓縮也就越強。
*/
Mat LogChange(Mat src);

/*
基於伽馬變換的圖像增強

伽馬變換主要用於圖像的校正，將灰度過高或者灰度過低的圖片進行修正，增強對比度。變換公式就是對原圖像上每一個像素值做乘積運算
伽馬變換對圖像的修正作用其實就是通過增強低灰度或高灰度的細節實現的
γ值以1為分界，值越小，對圖像低灰度部分的擴展作用就越強，值越大，對圖像高灰度部分的擴展作用就越強，通過不同的γ值，就可以達到增強低灰度或高灰度部分細節的作用。
伽馬變換對於圖像對比度偏低，並且整體亮度值偏高（對於於相機過曝）情況下的圖像增強效果明顯。
*/
Mat GammaChange(Mat src);

void drawDice1(vector<int> &diceType, vector<Vec3f> circles);
void drawDice2(vector<int> &diceType, vector<Vec3f> circles);
void drawDice3(vector<int> &diceType, vector<Vec3f> circles);
void drawDice4(vector<int> &diceType, vector<Vec3f> circles);
void drawDice5(vector<int> &diceType, vector<Vec3f> circles);
void drawDice6(vector<int> &diceType, vector<Vec3f> circles);

void checkDicePoint(vector<Point> points);
int checkPointInDice(int x, int y);
int checkPointInDice(Point point);

vector<Vec3f> totlecircles;
vector<Vec3f> showCircles;
vector<Vec4i> totlelinesP;
vector<Vec4i> showLinep;
vector<Point> points;
vector<int> diceType;
Rect diceRect[3];				//儲存圖片比對出來的三顆骰子位置

Mat frame;
Mat frame1;
Mat framegray;
Mat mat;
Mat diceImage[24];
Mat dicebase[2];
Mat frame_watershed;

int nLastTime = 0;
int nNowTime = 0;
int nShowTime = 0;
int nLoop;
BOOL bDebug;

void watershedMat(Mat src_mat, Mat &dst_mat);

class WatershedSegmenter {
private:
	Mat markers;
public:
	void setMarkers(Mat& markerImage)
	{
		markerImage.convertTo(markers, CV_32S);
	}

	Mat process(Mat &image)
	{
		watershed(image, markers);
		markers.convertTo(markers, CV_8U);
		return markers;
	}
};
