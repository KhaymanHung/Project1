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

//�l����
double angle(int x1, int y1, int x2, int y2, int x3, int y3);

//���I�Z��
int dist(int x1, int y1, int x2, int y2);
int dist(Point a, Point b);

void checkDicePoint(Mat image);
void matchDices(int method);
void matchImage(int method);
double basicMatch(Mat image, Mat tepl, Point &point, int method);
double grayMatch(Mat image, Mat tepl, Point &point, int method);
double binaryMatch(Mat image, Mat tepl, Point &point, int method);

/*
��󪽤�ϧ��Ťƪ��Ϲ��W�j

����ϧ��ŤƬO�q�L�վ�Ϲ����Ƕ������A�ϱo�b0~255�Ƕ��W��������[���šA�����F�Ϲ������סA�F��ﵽ�Ϲ��D�[��ı�ĪG���ت��C���׸��C���Ϲ��A�X�ϥΪ���ϧ��ŤƤ�k�ӼW�j�Ϲ��Ӹ`�C
*/
Mat RGBChange(Mat src);

/*
���Դ��Դ���l���Ϲ��W�j

�ϥΤ��߬�5��8�F��Դ��Դ���l�P�Ϲ����n�i�H�F��U�ƼW�j�Ϲ����ت��C
�Դ��Դ���l�i�H�W�j�������Ϲ�����
*/
Mat EnhanceChange(Mat src);

/*
�����Log�ܴ����Ϲ��W�j

����ܴ��i�H�N�Ϲ����C�ǫ׭ȳ����X�i�A��ܥX�C�ǫ׳�����h���Ӹ`�A�N�䰪�ǫ׭ȳ������Y�A��ְ��ǫ׭ȳ������Ӹ`�A�q�ӹF��j�չϹ��C�ǫ׳������ت��C
��󤣦P�����ơA���ƶV�j�A��C�ǫ׳������X�i�N�V�j�A�ﰪ�ǫ׳��������Y�]�N�V�j�C
*/
Mat LogChange(Mat src);

/*
�������ܴ����Ϲ��W�j

�����ܴ��D�n�Ω�Ϲ����ե��A�N�ǫ׹L���Ϊ̦ǫ׹L�C���Ϥ��i��ץ��A�W�j���סC�ܴ������N�O���Ϲ��W�C�@�ӹ����Ȱ����n�B��
�����ܴ���Ϲ����ץ��@�Ψ��N�O�q�L�W�j�C�ǫשΰ��ǫת��Ӹ`��{��
�^�ȥH1�����ɡA�ȶV�p�A��Ϲ��C�ǫ׳������X�i�@�δN�V�j�A�ȶV�j�A��Ϲ����ǫ׳������X�i�@�δN�V�j�A�q�L���P���^�ȡA�N�i�H�F��W�j�C�ǫשΰ��ǫ׳����Ӹ`���@�ΡC
�����ܴ����Ϲ����װ��C�A�åB����G�׭Ȱ����]����۾��L�n�^���p�U���Ϲ��W�j�ĪG����C
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
Rect diceRect[3];				//�x�s�Ϥ����X�Ӫ��T����l��m

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
