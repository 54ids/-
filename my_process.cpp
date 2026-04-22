#include "my_process.h"
#include <opencv2/dnn.hpp>
#include <stdio.h>
#include <string.h>
using namespace cv;
using namespace std;

/*                                   操作说明！！！！！！！！！！！！！！！！
		只有按键的AD可以用，D往后一帧，A向前一帧，esc健退出
		鼠标'<'跳帧，鼠标'+'跳第一帧，鼠标'>'向前跳帧
		mode 0 本地模式，mode1图像仿真模式
		address 文件地址
		i     想要的第几帧
		showph 0 拟线图不显示处理后图像，1，拟线图显示处理后图像
		有bug请反馈
																												@垃圾之首
																												
*/
uint mode = 1;
bool showph = 0;
bool gray_mode = 1;//1灰度模式，0彩图模式
#define address "C:/Users/用户名/OneDrive/Desktop/5月图片/2024_04_19_17_12_17_Vedio.mp4"
int sum_fp = -1;//设定总帧数，避免不必要的帧
int L_black[120]={0}, R_black[120], M_black[120];
uint scc8660_image[120][160] = {0};  //图像原始数据
uint image_copy[120][160] = { 0 };//处理后的图像数据
uchar phR[120][160] = { 0 };
uchar phG[120][160] = { 0 };
uchar phB[120][160] = { 0 };
float phH[120][160] = { 0 }; 
float phS[120][160] = { 0 };
float phV[120][160] = { 0 };
uchar gray[120][160] = { 0 };
uchar cpy_gray[120][160] = { 0 };
uchar cpy_phR[120][160] = { 0 };
uchar cpy_phG[120][160] = { 0 };
uchar cpy_phB[120][160] = { 0 };
uint i =1;		//定义直接跳到该帧
Mat output;Mat frame;Mat test;char fps[8],ph_B[8],ph_G[8],ph_R[8];
/**********************************************此处编写仿真程序*******************************************************/
#define start_line   20
#define uint16 uint
#define Width 160
#define High  120
#define zera_position 30
float parameterB, parameterA, parameterx, parameterk, parameterb;
#define byte uchar
#define uint8 uchar
uint8 cross_left_down_flag = 0, cross_right_down_flag = 0, cross_left_up_flag = 0, cross_right_up_flag = 0;
uint8 cross_left_down_line = 0, cross_left_up_line = 0, cross_right_down_line = 0, cross_right_up_line = 0;
uint8 right_losecountm = 0, left_losecountm = 0;
#define left 1
#define right 2
uint8 scan_endline = 20;
uint8 last_left_up = 0, last_right_up = 0, last_left_down = 0, last_right_down = 0;
uint8 verti_L_black[Width], verti_R_black[Width], verti_M_black[Width];
uint8 road_width[High];
uint16 Lroad_width[High], Rroad_width[High];
uint8 circle_down_spot = 0;
uint8 circle_mid_spot = 0;
uint8 circle_up_spot = 0;
uint8 circle_type = 0;
uint8 circle_state = 0;
uint8 circle_fork = 0;
uint8 circle_leave_spot = 0;
uint8 barrier = 0;
uint8 barrier_type = 0;
uint8 barrier_down = 0, barrier_up = 0;
uint8 zera_flag = 0;
uint8 start = 2;
uint8 sideways_left = 0,sideways_right = 0;
uint8 right_card_flag = 0, left_card_flag = 0;
uint8 cross_flag = 0, cross_state = 0,cross_count=0;
uint8 sideways_type = 0;
uint8 position_cross[20] = {0,0,0,0,0 };
uint8 turn_direction = 1;
uint8 cross_flag_count = 0;
uint8 temp_position = 0;
uint8 small_circle_flag = 0;
uint8 s_curve_flag = 0;
int index = 0;
uint8 max_list = 0;
bool curve_flag = 0;
#define thres 120
#define st 30
int my_abs(int a, int b);
void fix_line(uint8 type, uint8 begin_line, uint8 end_line)
{
	uint8 i;
	if (type == left)
	{
		for (i = begin_line-1; i >= end_line+1; i--)
		{
			if (my_abs(L_black[i], L_black[i - 1]) >= 3 && my_abs(L_black[i], L_black[i + 1]) >= 3)L_black[i] = (L_black[i - 1] + L_black[i + 1]) / 2;
		}
	}
	if (type == right)
	{
		for (i = begin_line-1; i >= end_line+1; i--)
		{
			if (my_abs(R_black[i], R_black[i - 1]) >= 3 && my_abs(R_black[i], R_black[i + 1]) >= 3)R_black[i] = (R_black[i - 1] + R_black[i + 1]) / 2;
		}
	}
	if (type == 3)
	{
		for (i = begin_line-1; i >= end_line+1; i--)
		{
			if (my_abs(M_black[i], M_black[i - 1]) >= 3 && my_abs(M_black[i], M_black[i + 1]) >= 3)M_black[i] = (L_black[i - 1] + L_black[i + 1]) / 2;
		}
	}
}
//
// Created by RUPC on 2022/9/20.
//
#define RESULT_ROW 120//结果图行列
#define RESULT_COL 160
#define         USED_ROW                120  //用于透视图的行列
#define         USED_COL                160
#define PER_IMG     cpy_gray//SimBinImage:用于透视变换的图像
#define ImageUsed   *PerImg_ip//*PerImg_ip定义使用的图像，ImageUsed为用于巡线和识别的图像
typedef unsigned char       uint8_t;                                              // 无符号  8 bits
uint8_t* PerImg_ip[RESULT_ROW][RESULT_COL];
void ImagePerspective_Init(void) {

	static uint8_t BlackColor = 0;
	double change_un_Mat[3][3] = { {0.420220,0.268539,-31.429828},{-0.014614,-0.086386,20.566809},{-0.000385,0.003583,0.061004} };
	for (int i = 0; i < RESULT_COL; i++) {
		for (int j = 0; j < RESULT_ROW; j++) {
			int local_x = (int)((change_un_Mat[0][0] * i
				+ change_un_Mat[0][1] * j + change_un_Mat[0][2])
				/ (change_un_Mat[2][0] * i + change_un_Mat[2][1] * j
					+ change_un_Mat[2][2]));
			int local_y = (int)((change_un_Mat[1][0] * i
				+ change_un_Mat[1][1] * j + change_un_Mat[1][2])
				/ (change_un_Mat[2][0] * i + change_un_Mat[2][1] * j
					+ change_un_Mat[2][2]));
			if (local_x
				>= 0 && local_y >= 0 && local_y < USED_ROW && local_x < USED_COL) {
				PerImg_ip[j][i] = &PER_IMG[local_y][local_x];
			}
			else {
				PerImg_ip[j][i] = &BlackColor;          //&PER_IMG[0][0];
			}

		}
	}

}
/*完成摄像头初始化后，调用一次ImagePerspective_Init，此后，直接调用ImageUsed   即为透视结果*/
float retmax(float a, float b, float c)//求最大值
{
	float max = 0;
	max = a;
	if (max < b)
		max = b;
	if (max < c)
		max = c;
	return max;
}
float retmin(float a, float b, float c)//求最小值
{
	float min = 0;
	min = a;
	if (min > b)
		min = b;
	if (min > c)
		min = c;
	return min;
}
//R,G,B参数传入范围（0~100）
//转换结果h(0~360),s(0~100),v(0~100)
void find_max_list()
{
	int i, j, max = 0;uint8 index = 0;
	for (i = 0; i < Width; i++)
	{
		index = 0;
		for (j = 0; j < High; j++)
		{
			if (gray[j][i])
			{
				index++;
			}
			else
			{
				if (index >= max)
				{
					max = index;
					max_list = i;
				}
				index = 0;
			}
		}
		if (index >= max)
		{
			max = index;
			max_list = i;
		}
	}
	left_losecountm = 0; right_losecountm = 0;
	for (i = 118; i >= 0; i--)
	{
		for (j = max_list; j > 2; j--)
		{
			if (gray[i][j] != 0 && gray[i][j - 1] == 0 && gray[i][j - 2] == 0)
			{
				L_black[i] = j;
				break;
			}
		}
		if (j <= 2)
		{
			L_black[i] = 0;
			left_losecountm++;
		}
		if (L_black[i] < 5)left_losecountm++;
		for (j = max_list; j < Width - 2; j++)
		{
			if (gray[i][j] != 0 && gray[i][j + 1] == 0 && gray[i][j + 2] == 0)
			{
				R_black[i] = j;
				break;
			}
		}
		if (j >= Width - 2)
		{
			R_black[i] = Width - 1;
			right_losecountm++;
		}
		if (R_black[i] > Width - 5)right_losecountm++;
		M_black[i] = (L_black[i] + R_black[i]) / 2;
		road_width[i] = R_black[i] - L_black[i];
	}
}
void rgb_to_hsv(float* h, float* s, float* v, float R, float G, float B)
{
	float max = 0, min = 0;
	R = R / 100;
	G = G / 100;
	B = B / 100;

	max = retmax(R, G, B);
	min = retmin(R, G, B);
	*v = max;
	if (max == 0)
		*s = 0;
	else
		*s = 1 - (min / max);

	if (max == min)
		*h = 0;
	else if (max == R && G >= B)
		*h = 60 * ((G - B) / (max - min));
	else if (max == R && G < B)
		*h = 60 * ((G - B) / (max - min)) + 360;
	else if (max == G)
		*h = 60 * ((B - R) / (max - min)) + 120;
	else if (max == B)
		*h = 60 * ((R - G) / (max - min)) + 240;

	*v = *v * 100;
	*s = *s * 100;
}
int limit_image_line(int line)
{
	if (line <= 0) line = 0;
	else if (line >= High - 1) line = High - 1;
	return line;
}
int limit_image_list(int list)
{
	if (list <= 0) list = 0;
	else if (list >= Width) list = Width - 1;
	return list;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       my_Bin_Image_Filter()
// @param       void
// @return      voidm
// @function    噪点滤波
//-------------------------------------------------------------------------------------------------------------------
void my_Bin_Image_Filter()
{
	byte nr; //行
	byte nc; //列
	for (nr = 1; nr < High - 1; nr++)
	{
		for (nc = 1; nc < Width - 1; nc = (byte)(nc + 1))
		{
			if ((gray[nr][nc] == 0)
				&& (gray[nr - 1][nc] + gray[nr + 1][nc] + gray[nr][nc + 1] + gray[nr][nc - 1] > index*2 ))
			{
				gray[nr][nc] = 0xff;
			}
			else if ((gray[nr][nc] != 0)
				&& (gray[nr - 1][nc] + gray[nr + 1][nc] + gray[nr][nc + 1] + gray[nr][nc - 1] < index*2))
			{
				gray[nr][nc] = 0;
			}
		}
	}
}
/*
* 获取RGB值
*/
void get_RGB( )
{
	int i, j;
	for (i = 0; i < 120; i++)
	{
		for (j = 0; j < 160; j++)
		{
			phB[i][j] = scc8660_image[i][j];
			phG[i][j] = scc8660_image[i][j]>>8;
			phR[i][j] = scc8660_image[i][j]>>16;
			//rgb_to_hsv(&phH[i][j], &phS[i][j], &phV[i][j], phR[i][j], phG[i][j], phB[i][j]);
		}
	}
}
/*
 * 图像边缘检测
 */
void Sobel(uchar* pSource, uchar* pDst)
{
	int             i, j, Gx, Gy, nSum;
	uchar* pDs1, * pDs2, * pDs3, * pDs4, * pDst5, * pDst6, * pDst7, * pDst8, * pDst9, * pResut;
	//memset(pDst,0,sizeof(unsigned char)*Width*High);
	pDs1 = pSource;
	pDs2 = pDs1 + 1;
	pDs3 = pDs2 + 1;
	pDs4 = pSource + Width;
	pDst5 = pDs4 + 1;
	pDst6 = pDst5 + 1;
	pDst7 = pSource + 2 * Width;
	pDst8 = pDst7 + 1;
	pDst9 = pDst8 + 1;
	pResut = pDst + Width + 1;
	for (i = 1; i < High-1; i++)
	{
		for (j = 1; j < Width - 1; j++)
		{
			Gx = (*pDs3) + 2 * (*pDst6) + (*pDst9)
				- (*pDst7) - (*pDs1) - 2 * (*pDs4);
			Gy = (*pDs1) + 2 * (*pDs2) + (*pDs3)
				- (*pDst7) - 2 * (*pDst8) - (*pDst9);
			if (Gx < 0)Gx = -Gx;
			if (Gy < 0)Gy = -Gy;
			nSum = Gx + Gy;
			if (nSum > 175)*pResut = 255;
			else *pResut = *pSource;
			pDs1++;
			pDs2++;
			pDs3++;
			pDs4++;
			pDst5++;
			pDst6++;
			pDst7++;
			pDst8++;
			pDst9++;
			pResut++;
		}
		pDs1 += 2;
		pDs2 += 2;
		pDs3 += 2;
		pDs4 += 2;
		pDst5 += 2;
		pDst6 += 2;
		pDst7 += 2;
		pDst8 += 2;
		pDst9 += 2;
		pResut += 2;
	}
}
/*
		最大值滤波
*/
void max_filter(uint16 pSource[High][Width], uint16 pDst[High][Width])
{
	int i, j;
	uint16 max = 0;
	for (i = 1; i < 120 - 1; i++)
	{
		for (j = 1; j < 160 - 1; j++)
		{
			if (pSource[i - 1][j-1] >= max)max = pSource[i - 1][j-1];
			if (pSource[i - 1][j] >= max)max = pSource[i - 1][j];
			if (pSource[i - 1][j+1] >= max)max = pSource[i - 1][j+1];
			if (pSource[i][j - 1] >= max)max = pSource[i][j - 1];
			if (pSource[i][j + 1] >= max)max = pSource[i][j + 1];
			if (pSource[i + 1][j - 1] >= max)max = pSource[i + 1][j - 1];
			if (pSource[i + 1][j] >= max)max = pSource[i + 1][j];
			if (pSource[i + 1][j + 1] >= max)max = pSource[i + 1][j + 1];
			if (pSource[i][j] >= max)max = pSource[i][j];
			pDst[i][j] = max;
			max = 0;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       turn_gray( )-转灰度
// @param       void
// @return      void
// @function    RGB转灰度
//-------------------------------------------------------------------------------------------------------------------
void turn_gray( )
{
	int i, j;
	for (i = 0; i <120; i++)
	{
		for (j = 0; j <160; j++)
		{
			gray[i][j] = phR[i][j] * 0.299 + phG[i][j] * 0.587 + phB[i][j] * 0.114;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       my_abs
// @param       void
// @return      void
// @function    int求绝对值
//-------------------------------------------------------------------------------------------------------------------
int my_abs(int a, int b)
{
	if (a > b)return (a - b);
	else return (b - a);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       base_sort()-基础图象函数
// @param       void
// @return      void
// @function    基础扫线整理
//-------------------------------------------------------------------------------------------------------------------
void base_sort()
{
	for (int line = High-2; line >=0; line--)
	{
		
		/*if (R_black[line] - L_black[line] < 5)
		{
			L_black[line] = L_black[line+1];
			R_black[line] = R_black[line + 1];
		}*/
		M_black[line] = (L_black[line] / 2 + R_black[line] / 2);
		if (line <High-1) //从第二行开始权重找中线
		{
			M_black[line] = (0.25 * M_black[line + 1] + 0.75 * M_black[line]);//权重比计算新中线
			//while (my_abs(M_black[line], M_black[line + 1]) > 5)//修补中线
			//{
			//	M_black[line] = (M_black[line] / 2 + M_black[line + 1] / 2);
			//}
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       advanced_regression()-图像辅助处理函数
// @param       void
// @return      void
// @function    通过两段直线计算拟合直线参数   0-中线    1-左线   2-右线
//-------------------------------------------------------------------------------------------------------------------
void advanced_regression(int type, int startline1, int endline1, int startline2, int endline2)
{
	startline1 = limit_image_line(startline1);
	startline2 = limit_image_line(startline2);
	endline1 = limit_image_line(endline1);
	endline2 = limit_image_line(endline2);
	fix_line(type, startline1, endline1);
	fix_line(type, startline2, endline2);
	int i = 0;
	int sumlines1 =startline1 -endline1;
	int sumlines2 = startline2-endline2;
	int sumX = 0;
	int sumY = 0;
	float averageX = 0;
	float averageY = 0;
	float sumUp = 0;
	float sumDown = 0;
	if (type == 0)  //拟合中线
	{
		/**计算sumX sumY**/
		for (i = endline1; i < startline1; i++)
		{
			sumX += i;
			sumY += M_black[i];
		}
		for (i = endline2; i < startline2; i++)
		{
			sumX += i;
			sumY += M_black[i];
		}
		averageX = (float)(sumX / (sumlines1 + sumlines2));     //x的平均值
		averageY = (float)(sumY / (sumlines1 + sumlines2));     //y的平均值
		for (i = endline1; i < startline1; i++)
		{
			sumUp += (M_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		for (i = endline2; i < startline2; i++)
		{
			sumUp += (M_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 1)     //拟合左线
	{
		/**计算sumX sumY**/
		for (i = endline1; i < startline1; i++)
		{
			sumX += i;
			sumY += L_black[i];
		}
		for (i = endline2; i < startline2; i++)
		{
			sumX += i;
			sumY += L_black[i];
		}
		averageX = (float)(sumX*1.0/ (sumlines1 + sumlines2));     //x的平均值
		averageY = (float)(sumY*1.0 / (sumlines1 + sumlines2));     //y的平均值
		for (i = endline1; i < startline1; i++)
		{
			sumUp += (L_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		for (i = endline2; i < startline2; i++)
		{
			sumUp += (L_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 2)         //拟合右线
	{
		/**计算sumX sumY**/
		for (i = endline1; i < startline1; i++)
		{
			sumX += i;
			sumY += R_black[i];
		}
		for (i = endline2; i < startline2; i++)
		{
			sumX += i;
			sumY += R_black[i];
		}
		averageX = (float)(sumX / (sumlines1 + sumlines2));     //x的平均值
		averageY = (float)(sumY / (sumlines1 + sumlines2));     //y的平均值
		for (i = endline1; i < startline1; i++)
		{
			sumUp += (R_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		for (i = endline2; i < startline2; i++)
		{
			sumUp += (R_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       regression()-图像辅助处理函数
// @param       void
// @return      void
// @function    通过一段直线计算拟合直线参数   0-中线    1-左线   2-右线 3-镜像左线 4-镜像右线
//-------------------------------------------------------------------------------------------------------------------
void regression(int type, int startline, int endline)//最小二乘法拟合曲线，分别拟合中线，左线，右线,type表示拟合哪几条线   xy 颠倒
{
	startline = limit_image_line(startline);
	endline = limit_image_line(endline);
	fix_line(type, startline, endline);
	int i = 0;
	int sumlines =  startline-endline;
	int sumX = 0;
	int sumY = 0;
	float averageX = 0;
	float averageY = 0;
	float sumUp = 0;
	float sumDown = 0;
	if (type == 0)      //拟合中线
	{
		for (i = endline; i < startline; i++)
		{
			sumX += i;
			sumY += M_black[i];
		}
		if (sumlines != 0)
		{
			averageX = (float)(sumX / sumlines);     //x的平均值
			averageY = (float)(sumY / sumlines);     //y的平均值
		}
		else
		{
			averageX = 0;     //x的平均值
			averageY = 0;     //y的平均值
		}
		for (i = endline; i < startline; i++)
		{
			sumUp += (M_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 1)//拟合左线
	{
		for (i = endline; i < startline; i++)
		{
			sumX += i;
			sumY += L_black[i];
		}
		if (sumlines == 0) sumlines = 1;
		averageX = (float)(sumX / sumlines);     //x的平均值
		averageY = (float)(sumY / sumlines);     //y的平均值
		for (i = endline; i < startline; i++)
		{
			sumUp += (L_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 2)//拟合右线
	{
		for (i = endline; i < startline; i++)
		{
			sumX += i;
			sumY += R_black[i];
		}
		if (sumlines == 0) sumlines = 1;
		averageX = (float)(sumX / sumlines);     //x的平均值
		averageY = (float)(sumY / sumlines);     //y的平均值
		for (i = endline; i < startline; i++)
		{
			sumUp += (R_black[i] - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 3)
	{
		for (i = endline; i < startline; i++)
		{
			sumX += i;
			sumY += (159-R_black[i]);
		}
		if (sumlines == 0) sumlines = 1;
		averageX = (float)(sumX / sumlines);     //x的平均值
		averageY = (float)(sumY / sumlines);     //y的平均值
		for (i = endline; i < startline; i++)
		{
			sumUp += ((159-R_black[i]) - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
	else if (type == 4)
	{
		for (i = endline; i < startline; i++)
		{
			sumX += i;
			sumY += (159 - L_black[i]);
		}
		if (sumlines == 0) sumlines = 1;
		averageX = (float)(sumX / sumlines);     //x的平均值
		averageY = (float)(sumY / sumlines);     //y的平均值
		for (i = endline; i < startline; i++)
		{
			sumUp += ((159 - L_black[i]) - averageY) * (i - averageX);
			sumDown += (i - averageX) * (i - averageX);
		}
		if (sumDown == 0) parameterB = 0;
		else parameterB = sumUp / sumDown;
		parameterA = averageY - parameterB * averageX;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       two_point_regression()-图像辅助处理函数
// @param       void
// @return      void
// @function    通过两段直线计算拟合直线参数
//-------------------------------------------------------------------------------------------------------------------
void two_point_regression(int y2, int x2, int y1, int x1)
{
	y2 = limit_image_line(y2);
	x2 = limit_image_list(x2);
	y1 = limit_image_line(y1);
	x1 = limit_image_list(x1);
	if (x1 == x2) parameterx = x1;
	else
	{
		parameterk = (float)(y1 - y2) / (float)(x1 - x2);
		parameterb = (float)((float)(y1 - parameterk * x1) + (float)(y2 - parameterk * x2)) / 2;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       draw_two_point()-图像辅助处理函数
// @param       void
// @return      void
// @function    根据两行间斜率和截距拟合直线   1-左线  2-右线
//-------------------------------------------------------------------------------------------------------------------
void draw_two_point(int type, int start_y, int stop_y, float parameterk, float parameterb)
{
	int x;
	start_y = limit_image_line(start_y);
	stop_y = limit_image_line(stop_y);
	if (type == 1)//拉左线
	{
		for (int i = stop_y; i <= start_y; i++)
		{
			if (parameterx == 0)
			{
				x = (int)((i - parameterb) / parameterk);
				x=limit_image_list(x);
			}
			else x = (int)parameterx;
			L_black[i] = (byte)x;
		}
	}
	if (type == 2)//拉右线
	{
		for (int i = stop_y; i <= start_y; i++)
		{
			if (parameterx == 0)
			{
				x = (int)((i - parameterb) / parameterk);
				x = limit_image_list(x);
			}
			else x = (int)parameterx;
			R_black[i] = (byte)x;
		}
	}
	if (type == 3)//拉中线
	{
		for (int i = stop_y; i <= start_y; i++)
		{
			if (parameterx == 0)
			{
				x = (int)((i - parameterb) / parameterk);
				x = limit_image_list(x);
			}
			else x = (int)parameterx;
			M_black[i] = (byte)x;
		}
	}
	parameterx = 0;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       my_find_baseline()-基础图像函数
// @param       寻找的行
// @return      void
// @function    寻找基准线
//-------------------------------------------------------------------------------------------------------------------
void my_find_baseline(int line)
{
	byte j;
	byte s_line = 0, e_line = 0;
	byte s_linem = 0, e_linem = 0;
	byte flag = 0;
	for (j = 0; j < Width-1; j++)
	{
		if (gray[line][j] != 0)
		{
			if (flag == 0)
			{
				s_line = j;
				flag = 1;
			}
			e_line = j;

		}
		else flag = 0;
		if (e_line - s_line > e_linem - s_linem)
		{
			s_linem = s_line;
			e_linem = e_line;
		}
	}
	if (line == 0)
	{
		s_linem = s_linem <= 0 ? (byte)0 : s_linem;
		e_linem = e_linem >= 159 ? (byte)159 : e_linem;
	}
	L_black[line] = s_linem;
	R_black[line] = e_linem;
	M_black[line] = (byte)(e_linem / 2 + s_linem / 2);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       find_line()-基础扫线函数
// @param       void
// @return      void
// @function    寻找左右线
//-------------------------------------------------------------------------------------------------------------------
void find_line()
{
	my_find_baseline(119);
	int i, j;
	uchar l_flag, r_flag;
	left_losecountm = 0; right_losecountm = 0;
	scan_endline = 0;
	uint8 white_flag = 0;
	for (i = 118; i >= 0; i--)
	{
		white_flag = 0;
		for (j = max_list; j > 2; j--)
		{
			if (gray[i][j])white_flag = 1;
			if (gray[i][j] != 0&& gray[i][j-1]==0&& gray[i][j-2]==0)
			{
				L_black[i] = j;
				l_flag = 1;
				break;
			}
		}
		if (j <= 2)
		{
			L_black[i] = 0;
			left_losecountm++;
			l_flag = 0;
		}
		if (L_black[i] < 5)left_losecountm++;
		for (j = max_list + 1; j < Width-2; j++)
		{
			if (gray[i][j])white_flag = 1;
			if (gray[i][j] != 0&& gray[i][j+1]==0&& gray[i][j+2]==0)
			{
				R_black[i] =j;
				r_flag = 1;
				break;
			}
		}
		if (j >= Width - 2)
		{
			R_black[i] = Width-1;
			right_losecountm++;
			r_flag = 0;
		}
		if (R_black[i] > Width - 5)right_losecountm++;
		if (l_flag == 0 && r_flag == 0)
		{
			my_find_baseline(i);
		}
		else if((l_flag == 0 || r_flag == 0)&&i>=30)my_find_baseline(i);
		else M_black[i] = (L_black[i] + R_black[i]) / 2;
		road_width[i] = R_black[i+1]-L_black[i];
		if (white_flag == 0&&!scan_endline)scan_endline = i;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       环岛寻线_find_line()-基础扫线函数
// @param       void
// @return      void
// @function    寻找左右线
//-------------------------------------------------------------------------------------------------------------------
void circle_find_line()
{
	my_find_baseline(119);
	int i, j;
	uchar l_flag, r_flag;
	left_losecountm = 0; right_losecountm = 0;
	scan_endline = 0;
	uint8 white_flag = 0;
	for (i = 118; i >= 0; i--)
	{
		white_flag = 0;
		for (j = M_black[i + 1]; j > 2; j--)
		{
			if (gray[i][j])white_flag = 1;
			if (gray[i][j] != 0 && gray[i][j - 1] == 0 && gray[i][j - 2] == 0)
			{
				L_black[i] = j;
				l_flag = 1;
				break;
			}
		}
		if (j <= 2)
		{
			L_black[i] = 0;
			left_losecountm++;
			l_flag = 0;
		}
		if (L_black[i] < 5)left_losecountm++;
		for (j = M_black[i + 1] + 1; j < Width - 2; j++)
		{
			if (gray[i][j])white_flag = 1;
			if (gray[i][j] != 0 && gray[i][j + 1] == 0 && gray[i][j + 2] == 0)
			{
				R_black[i] = j;
				r_flag = 1;
				break;
			}
		}
		if (j >= Width - 2)
		{
			R_black[i] = Width - 1;
			right_losecountm++;
			r_flag = 0;
		}
		if (R_black[i] > Width - 5)right_losecountm++;
		if (l_flag == 0 || r_flag == 0)
		{
			my_find_baseline(i);
		}
		else M_black[i] = (L_black[i] + R_black[i]) / 2;
		road_width[i] = R_black[i + 1] - L_black[i];
		if (white_flag == 0&&!scan_endline)scan_endline = i;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       car_protect()
// @param       void
// @return      0xff 未出线 0 出线
// @function    小车出线保护
//-------------------------------------------------------------------------------------------------------------------
byte car_protect()
{
	byte i, j;
	byte flag = 0xff;
	for (i = High-1; i >= High-3; i--)
	{
		for (j = 10; j < Width-10; j++)
		{
			if (cpy_gray[i][j] != 0)
			{
				flag = 0;
				break;
			}
		}
	}
	return flag;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       dst_find_white()-分割白色赛道
// @param       void
// @return      void
// @function    分割白色赛道
//-------------------------------------------------------------------------------------------------------------------
void dst_find_white()
{
	int i, j;
	for (i = 0; i < 120; i++)
	{
		for (j = 0; j < 160; j++)
		{
			/*if ((phB[i][j] > 160 && phG[i][j] > 160 && phR[i][j] > 160 && my_abs(phB[i][j], phG[i][j]) < 70 && my_abs(phB[i][j], phR[i][j]) < 70 && my_abs(phR[i][j], phG[i][j]) < 70)
				|| (phB[i][j] > 130 && phG[i][j] > 130 && phR[i][j] > 130 && my_abs(phB[i][j], phG[i][j]) < 50 && my_abs(phB[i][j], phR[i][j]) < 50 && my_abs(phR[i][j], phG[i][j]) < 50)
				|| (phB[i][j] > 190 && phG[i][j] > 190 && phR[i][j] > 190))*/
			if ((phB[i][j] > thres && phG[i][j] > thres && phR[i][j] > thres && my_abs(phB[i][j], phG[i][j]) < st && my_abs(phB[i][j], phR[i][j]) < st && my_abs(phR[i][j], phG[i][j]) < st)
				|| (phB[i][j] > thres + st && phG[i][j] > thres + st && phR[i][j] > thres + st && my_abs(phB[i][j], phG[i][j]) < 70 && my_abs(phB[i][j], phR[i][j]) < 70 && my_abs(phR[i][j], phG[i][j]) < 70))
			/*if(phS[i][j]<=30&&phV[i][j] >= 96)*/
			{
				image_copy[i][j] = 0xffffff;
				gray[i][j] = (uchar)image_copy[i][j];
				cpy_gray[i][j] = (uchar)image_copy[i][j];
			}
			else
			{
				image_copy[i][j] = 0;
				cpy_gray[i][j] = (uchar)image_copy[i][j];
				gray[i][j] = (uchar)image_copy[i][j];
			}
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       draw_line()-图像辅助处理函数
// @param       void
// @return      void
// @function    根据斜率和截距拟合直线   0-中线   1-左线  2-右线
//-------------------------------------------------------------------------------------------------------------------
void draw_line(int type, int startline, int endline, float parameterB, float parameterA)
{
	int num;
	startline = limit_image_line(startline);
	endline = limit_image_line(endline);
	if (type == 1)              //左
	{
		for (int i = endline; i < startline; i++)
		{
			num = (int)(parameterB * i + parameterA);
			num=limit_image_list(num);
			L_black[i] = (byte)num;
		}
	}
	else if (type == 2)            //右
	{
		for (int i = endline; i < startline; i++)
		{
			num = (int)(parameterB * i + parameterA);
			num = limit_image_list(num);
			R_black[i] = (byte)num;
		}
	}
	else if (type == 0)             //中
	{
		for (int i = endline; i < startline; i++)
		{
			num = (int)(parameterB * i + parameterA);
			num = limit_image_list(num);
			M_black[i] = (byte)num;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       my_line_offset()-图像辅助处理函数
// @param       int
// @return      int
// @function    新方差 0-中线  1-左线   2-右线
//-------------------------------------------------------------------------------------------------------------------
int my_line_offset(byte type, byte startline, byte endline)
{
	int line_offset = 0;
	startline = limit_image_line(startline);
	endline = limit_image_line(endline);
	if (type == 0)
	{
		regression(0, startline, endline);
		for (byte h = endline; h < startline; h++)
		{
			line_offset += my_abs((byte)(parameterB * h + parameterA), M_black[h]);
		}
		return line_offset;
	}
	else if (type == 1)
	{
		regression(1, startline, endline);
		for (byte h = endline; h < startline; h++)
		{
			line_offset += my_abs((byte)(parameterB * h + parameterA), L_black[h]);
		}
		return line_offset;
	}
	else if (type == 2)
	{
		regression(2, startline, endline);
		for (byte h = endline; h < startline; h++)
		{
			line_offset += my_abs((byte)(parameterB * h + parameterA), R_black[h]);
		}
		return line_offset;
	}
	return 0;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       zongxiang_saoxian()
// @param       int
// @return      void
// @function    纵向扫基础线
//-------------------------------------------------------------------------------------------------------------------
void zongxiang_saoxian(int line)
{
	byte j;
	byte s_line = 0, e_line = 0;
	byte s_linem = 0, e_linem = 0;
	byte flag = 0;
	for (j = 0; j < High - 1; j++)
	{
		if (gray[j][line] != 0)
		{
			if (flag == 0)
			{
				s_line = j;
				flag = 1;
			}
			e_line = j;

		}
		else flag = 0;
		if (e_line - s_line > e_linem - s_linem)
		{
			s_linem = s_line;
			e_linem = e_line;
		}
	}
	if (line == 0)
	{
		s_linem = s_linem <= 0 ? (byte)0 : s_linem;
		e_linem = e_linem >= 159 ? (byte)159 : e_linem;
	}
	verti_L_black[line] = s_linem;
	verti_R_black[line] = e_linem;
	verti_M_black[line] = (s_linem + e_linem) / 2;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       zongxiang_saoxian(uint8 start_list,uint8 endlist)
// @param       uint8
// @return      void
// @function    纵向扫线
//-------------------------------------------------------------------------------------------------------------------
void zongxiang_saoxian(uint8 startlist,uint8 endlist)
{
	startlist = limit_image_list(startlist);
	endlist = limit_image_list(endlist);
	zongxiang_saoxian(startlist);
	int i, j;
	uchar l_flag=0, r_flag=0;
	for (i = startlist; i <endlist; i++)
	{
		for (j = verti_M_black[i - 1]; j > 2; j--)
		{
			if (gray[j][i] != 0 && gray[j-1][i] == 0 && gray[j-2][i] == 0)
			{
				verti_L_black[i] = j;
				l_flag = 1;
				break;
			}
		}
		if (j <= 2)
		{
			verti_L_black[i] = 0;
			l_flag = 0;
		}
		for (j = verti_M_black[i - 1] + 1; j < High- 2; j++)
		{
			if (gray[j][i] != 0 && gray[j+1][i] == 0 && gray[j+2][i] == 0)
			{
				verti_R_black[i] = j;
				r_flag = 1;
				break;
			}
		}
		if (j >= 160 - 2)
		{
			verti_R_black[i] = 159;
			r_flag = 0;
		}
		if (verti_R_black[i] > Width - 5)right_losecountm++;
		verti_M_black[i] = (verti_L_black[i] + verti_R_black[i]) / 2;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       check_cross_spot()
// @param       void
// @return      void
// @function    检查十字拐点
//-------------------------------------------------------------------------------------------------------------------
void check_cross_spot()
{
	if (cross_left_up_flag == 1)
	{
		if (cross_left_up_line+3 >= cross_left_down_line)
		{
			cross_left_down_flag = 0;
			cross_left_down_line = 0;
		}

	}
	if (cross_right_up_flag == 1)
	{
		if (cross_right_up_line+3 >= cross_right_down_line)
		{
			cross_right_down_flag = 0;
			cross_right_down_line = 0;
		}
	}
	if (cross_left_down_flag)
	{
		if (!sideways_left)
		{
			if (L_black[cross_left_down_line - 5] + 10 > L_black[cross_left_down_line])
			{
				cross_left_down_flag = 0;
			}
		}
	}
	if (cross_right_down_flag)
	{
		if (!sideways_right)
		{
			if (R_black[cross_right_down_line - 5] < R_black[cross_right_down_line] + 10)
			{
				cross_right_down_flag = 0;
			}
		}
	}
	if (cross_left_up_flag&& cross_right_up_flag)
	{
		if (cross_left_up_line+40 <= cross_right_up_line)
		{
			cross_left_up_flag = 0;
		}
		if (cross_right_up_line + 40 <= cross_left_up_line)
		{
			cross_right_up_flag = 0;
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       find_cross_spot()
// @param       void
// @return      void
// @function    寻找十字拐点
//-------------------------------------------------------------------------------------------------------------------
void find_cross_spot()
{
	cross_left_down_flag = 0; cross_right_down_flag = 0; cross_left_up_flag = 0; cross_right_up_flag = 0;
	cross_left_down_line = 0; cross_left_up_line = 0; cross_right_down_line = 0; cross_right_up_line = 0;
	sideways_left = 0; sideways_right = 0;
	int i;
	byte tempi = 0;
	for (i = High-start_line; i >=8; i--)
	{
		if(!road_width[i])continue;
		if (cross_left_down_flag == 0)
		{
			if (L_black[i] > 10 && L_black[i - 1] < 5 && L_black[i - 2] < 5 && L_black[i - 3] < 5 && L_black[i - 4] < 5)
			{
				for (tempi = i; tempi < i + 8; tempi++)
				{
					if (L_black[tempi] > 10 && my_abs(L_black[tempi], L_black[tempi + 1]) < 3 && my_abs(L_black[tempi + 1], L_black[tempi + 2]) < 3 && my_abs(L_black[tempi + 2], L_black[tempi + 3]) < 3
						&& L_black[tempi + 1] >= L_black[tempi + 3] && L_black[tempi + 2] >= L_black[tempi + 4])
					{
						cross_left_down_flag = 1;
						cross_left_down_line = tempi;
						break;
					}
				}
			}
			else if (L_black[i] > L_black[i - 3] && L_black[i] > L_black[i + 3] && L_black[i] > 10 && L_black[i - 1] > 10 && L_black[i + 1] > 10
				&& my_abs(L_black[i + 1], L_black[i + 2]) < 5 && my_abs(L_black[i + 2], L_black[i + 3]) < 5 && my_abs(L_black[i + 3], L_black[i + 4]) < 5
				&& my_abs(L_black[i - 1], L_black[i - 2]) < 10 && my_abs(L_black[i - 2], L_black[i - 3]) < 10 && my_abs(L_black[i - 3], L_black[i - 4]) < 10
				 && L_black[i + 1] >= L_black[i + 3] && L_black[i + 2] >= L_black[i + 4])
			{
				cross_left_down_flag = 1;
				cross_left_down_line = i;
				sideways_left = 1;
			}
		}
		if (cross_right_down_flag == 0)
		{
			if (R_black[i] <Width- 5 && R_black[i - 1] >Width- 5 && R_black[i - 2] >Width- 5 && R_black[i - 3] >Width- 5 && R_black[i - 4] >Width- 10)
			{
				for (tempi = i; tempi < i + 8; tempi++)
				{
					if (R_black[tempi] <Width- 10 && my_abs(R_black[tempi], R_black[tempi + 1]) < 3 && my_abs(R_black[tempi + 1], R_black[tempi + 2]) < 3 && my_abs(R_black[tempi + 2], R_black[tempi + 3]) < 3
						&& R_black[tempi + 1] <= R_black[tempi + 3] && R_black[tempi + 2] <= R_black[tempi + 4])
					{
						cross_right_down_flag = 1;
						cross_right_down_line = tempi;
						break;
					}
				}
			}
			else if (R_black[i] < R_black[i - 3] && R_black[i] < R_black[i + 3] && R_black[i] <Width- 10 && R_black[i - 1] <Width- 10 && R_black[i + 1] <Width- 10
				&& my_abs(R_black[i + 1], R_black[i + 2]) < 5 && my_abs(R_black[i + 2], R_black[i + 3]) < 5 && my_abs(R_black[i + 3], R_black[i + 4]) < 5
				&& my_abs(R_black[i - 1], R_black[i - 2]) < 10 && my_abs(R_black[i - 2], R_black[i - 3]) < 10 && my_abs(R_black[i - 3], R_black[i - 4]) < 10
				&& R_black[i + 1] <= R_black[i + 3] && R_black[i + 2] <= R_black[i + 4])
			{
				cross_right_down_flag = 1;
				cross_right_down_line = i;
				sideways_right = 1;
				printf("sideways_right");
			}
		}
		if (cross_left_up_flag == 0)
		{
			if (L_black[i + 3] < 5 && L_black[i + 2] < 5 && L_black[i + 1] < 5)
			{
				for (tempi = i; tempi >=i-5; tempi--)
				{
					if (L_black[tempi]<Width-15&&L_black[tempi]>10 && L_black[tempi - 1] > 5 && my_abs(L_black[tempi], L_black[tempi - 1]) < 3 && my_abs(L_black[tempi - 1], L_black[tempi - 2]) < 3 && my_abs(L_black[tempi - 2], L_black[tempi - 3]) < 3)
					{
						cross_left_up_line = (byte)(tempi);
						cross_left_up_flag = 1;
						break;
					}
				}
				if (cross_left_up_flag == 1)
				{
					if (L_black[cross_left_up_line]  < 5 + L_black[cross_left_up_line + 3] || cross_left_up_line <= 10)
					{
						cross_left_up_flag = 0;
					}
				}
			}
		}
		if (cross_right_up_flag == 0)
		{
			if (R_black[i + 3] > Width-5 && R_black[i + 2] > Width-5 && R_black[i + 1] > Width-5)
			{
				for (tempi = i; tempi >= i-5 ; tempi--)
				{
					if (R_black[tempi]>15&&R_black[tempi - 1] < Width-10 && my_abs(R_black[tempi], R_black[tempi - 1]) < 3 && my_abs(R_black[tempi - 1], R_black[tempi - 2]) < 3 && my_abs(R_black[tempi - 2], R_black[tempi - 3]) < 3)
					{
						cross_right_up_line = (byte)(tempi);
						cross_right_up_flag = 1;
						break;
					}
				}
				if (cross_right_up_flag == 1)
				{
					if (R_black[cross_right_up_line + 3] < 5 + R_black[cross_right_up_line]|| cross_right_up_line<=10)
					{
						cross_right_up_flag = 0;
					}
				}
			}
		}
		if (cross_left_up_flag == 1 && cross_right_up_flag == 1) break;
	}
	check_cross_spot();
	printf("lu:%d %d ld:%d %d ru:%d %d  rd:%d %d\n", cross_left_up_flag, cross_left_up_line, 
		cross_left_down_flag, cross_left_down_line, 
		cross_right_up_flag, cross_right_up_line, 
		cross_right_down_flag, cross_right_down_line
	);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       cross_type_process()
// @param       void
// @return      void
// @function    十字拐点补线   0-中线   1-左线  2-右线
//-------------------------------------------------------------------------------------------------------------------
void cross_type_process()
{
	if (cross_right_down_flag == 1 && cross_right_up_flag == 1 && cross_left_down_flag == 1 && cross_left_up_flag == 1)
	{
		advanced_regression(left, cross_left_down_line + 5, cross_left_down_line, cross_left_up_line, cross_left_up_line - 3);
		draw_line(left, cross_left_down_line + 3, cross_left_up_line - 3, parameterB, parameterA);
		advanced_regression(right, cross_right_down_line + 5, cross_right_down_line, cross_right_up_line, cross_right_up_line - 3);
		draw_line(right, cross_right_down_line + 3, cross_right_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 0 && cross_left_down_flag == 1 && cross_left_up_flag == 0)
	{
		regression(left, cross_left_down_line + 5, cross_left_down_line);
		draw_line(left, cross_left_down_line + 3, 0, parameterB, parameterA);
		regression(right, cross_right_down_line + 5, cross_right_down_line);
		draw_line(right, cross_right_down_line + 3, 0, parameterB, parameterA);
	}
	if (cross_right_down_flag == 0 && cross_right_up_flag == 1 && cross_left_down_flag == 0 && cross_left_up_flag == 1)
	{
		regression(left, cross_left_up_line, cross_left_up_line - 8);
		draw_line(left, High - 1, cross_left_up_line - 3, parameterB, parameterA);
		regression(right, cross_right_up_line, cross_right_up_line - 8);
		draw_line(right, High - 1, cross_right_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 1 && cross_left_down_flag == 1 && cross_left_up_flag == 0)
	{
		regression(left, cross_left_down_line + 5, cross_left_down_line);
		draw_line(left, cross_left_down_line + 3, 0, parameterB, parameterA);
		advanced_regression(right, cross_right_down_line + 5, cross_right_down_line, cross_right_up_line, cross_right_up_line - 5);
		draw_line(right, cross_right_down_line + 3, cross_right_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 1 && cross_left_down_flag == 0 && cross_left_up_flag == 1)
	{
		regression(left, cross_left_up_line, cross_left_up_line - 5);
		draw_line(left, High-1, cross_left_up_line - 3, parameterB, parameterA);
		advanced_regression(right, cross_right_down_line + 5, cross_right_down_line, cross_right_up_line, cross_right_up_line - 5);
		draw_line(right, cross_right_down_line + 3, cross_right_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 0 && cross_right_up_flag == 1 && cross_left_down_flag == 1 && cross_left_up_flag == 1)
	{
		regression(right, cross_right_up_line, cross_right_up_line - 5);
		draw_line(right, High-1, cross_right_up_line - 3, parameterB, parameterA);
		advanced_regression(left, cross_left_down_line + 5, cross_left_down_line, cross_left_up_line, cross_left_up_line - 5);
		draw_line(left, cross_left_down_line + 3, cross_left_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 0 && cross_left_down_flag == 1 && cross_left_up_flag == 1)
	{
		regression(right, cross_right_down_line + 8, cross_right_down_line + 2);
		draw_line(right, cross_right_down_line + 3, 0, parameterB, parameterA);
		advanced_regression(left, cross_left_down_line + 8, cross_left_down_line, cross_left_up_line, cross_left_up_line - 5);
		draw_line(left, cross_left_down_line + 3, cross_left_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 0 && cross_left_down_flag == 0 && cross_left_up_flag == 1&&my_abs(cross_right_down_line, cross_left_up_line)<20)
	{
		if (cross_right_down_line >= 50 && cross_left_up_line >= 40)
		{
			regression(left, cross_left_up_line - 3, cross_left_up_line - 8);
			draw_line(left, High - 1, cross_left_up_line - 3, parameterB, parameterA);
			regression(right, cross_right_down_line + 8, cross_right_down_line + 3);
			draw_line(right, cross_right_down_line + 3, 0, parameterB, parameterA);
		}
	}
	if (cross_right_down_flag == 0 && cross_right_up_flag == 1 && cross_left_down_flag == 1 && cross_left_up_flag == 0 && my_abs(cross_right_up_line, cross_left_down_line) < 20)
	{
		if (cross_right_up_line >= 40 && cross_left_down_line >= 50)
		{
			regression(right, cross_right_up_line - 3, cross_right_up_line - 8);
			draw_line(right, High - 1, cross_right_up_line - 3, parameterB, parameterA);
			regression(left, cross_left_down_line + 8, cross_left_down_line + 3);
			draw_line(left, cross_left_down_line + 3, 0, parameterB, parameterA);
		}
	}
	if (cross_right_down_flag == 0 && cross_right_up_flag == 0 && cross_left_down_flag == 1 && cross_left_up_flag == 1)
	{
		advanced_regression(left, cross_left_down_line + 8, cross_left_down_line, cross_left_up_line, cross_left_up_line - 8);
		draw_line(left, cross_left_down_line + 3, cross_left_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 1 && cross_left_down_flag == 0 && cross_left_up_flag == 0)
	{
		advanced_regression(right, cross_right_down_line + 8, cross_right_down_line, cross_right_up_line, cross_right_up_line - 8);
		draw_line(right, cross_right_down_line + 3, cross_right_up_line - 3, parameterB, parameterA);
	}
	if (cross_right_down_flag == 1 && cross_right_up_flag == 0 && cross_left_down_flag == 0)
	{
		if (cross_state >= 1)
		{
			regression(right, cross_right_down_line + 10, cross_right_down_line);
			draw_line(right, cross_right_down_line + 6, 0, parameterB, parameterA);
		}
		else if (cross_right_down_line<=High-40)
		{
			regression(right, cross_right_down_line + 10, cross_right_down_line);
			draw_line(right, cross_right_down_line + 6, 0, parameterB, parameterA);
		}
	}
	if (cross_left_down_flag == 1 && cross_left_up_flag == 0 && cross_right_down_flag == 0)
	{
		if (cross_state >= 1)
		{
			regression(left, cross_left_down_line + 10, cross_left_down_line);
			draw_line(left, cross_left_down_line + 6, 0, parameterB, parameterA);
		}
		else if (cross_left_down_line<=High-40)
		{
			regression(left, cross_left_down_line + 10, cross_left_down_line);
			draw_line(left, cross_left_down_line + 6, 0, parameterB, parameterA);
		}
	}
	if (cross_right_down_flag == 0 && cross_right_up_flag == 1 && cross_left_up_flag == 0)
	{
		regression(right, cross_right_up_line, cross_right_up_line - 5);
		if (cross_right_up_line >= High - 25)
		{
			draw_line(right, cross_right_up_line + 20, cross_right_up_line - 3, parameterB, parameterA);
		}
		else if (cross_right_up_line > High - 50)
		{
			draw_line(right, High - 1, cross_right_up_line - 3, parameterB, parameterA);
		}
		else if (cross_state >= 1)
		{
			draw_line(right, High - 1, cross_right_up_line - 3, parameterB, parameterA);
		}
	}
	if (cross_left_down_flag == 0 && cross_left_up_flag == 1 && cross_right_up_flag == 0)
	{
		regression(left, cross_left_up_line, cross_left_up_line - 5);
		if (cross_left_up_line >= High - 25)
		{
			draw_line(left, cross_left_up_line + 20, cross_left_up_line - 3, parameterB, parameterA);
		}
		else if (cross_left_up_line > High - 50)
		{
			draw_line(left, High - 1, cross_left_up_line - 3, parameterB, parameterA);
		}
		else if (cross_state >= 1)
		{
			draw_line(left, High - 1, cross_left_up_line - 3, parameterB, parameterA);
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       area_find()
// @param       void
// @return      void
// @function    最大连通域寻找
//-------------------------------------------------------------------------------------------------------------------
void find_area()
{
	int S[200] = { 0 };
	int labelSet[200] = { 0 };
	int max = 0;
	int p = 0;
	int m = 0;
	int label = 0;
	int i, j;
	memset(cpy_gray, 0, sizeof(unsigned char) * Width * High);
	for (i = 1; i <= 119; i++)
	{
		for (j = 1; j <= 159; j++)
		{
			//***边界情况***
			if (i == 0 && j == 0)//上边缘(含角点)
			{
				if (image_copy[i][j] != 0)
				{
					label++;
					cpy_gray[i][j] = label;
					labelSet[label] = label;
				}
			}
			else if (i == 0) {
				if (image_copy[i][j] != 0)
				{
					if (cpy_gray[i][j - 1] == 0)
					{
						label++;
						cpy_gray[i][j] = label;
						labelSet[label] = label;
					}
					else {
						cpy_gray[i][j] = cpy_gray[i][j - 1];
					}
				}
			}
			else if (j == 0) {
				if (image_copy[i][j] != 0)
				{
					if (cpy_gray[i - 1][j] == 0)
					{
						label++;
						cpy_gray[i][j] = label;
						labelSet[label] = label;
					}
					else {
						cpy_gray[i][j] = cpy_gray[i - 1][j];
					}
				}
			}
			//不是边界的点
			if (image_copy[i][j] != 0)
			{
				if (cpy_gray[i - 1][j] == 0 && cpy_gray[i][j - 1] == 0)
				{
					label++;
					cpy_gray[i][j] = label;
					labelSet[cpy_gray[i][j]] = label;
				}
				else if (cpy_gray[i - 1][j] != 0 && cpy_gray[i][j - 1] == 0)
				{
					cpy_gray[i][j] = cpy_gray[i - 1][j];
				}
				else if (cpy_gray[i - 1][j] == 0 && cpy_gray[i][j - 1] != 0)
				{

					cpy_gray[i][j] = cpy_gray[i][j - 1];
				}
				else {
					if (cpy_gray[i][j - 1] > cpy_gray[i - 1][j])
					{
						labelSet[cpy_gray[i][j - 1]] = cpy_gray[i - 1][j];
						cpy_gray[i][j] = cpy_gray[i - 1][j];
					}
					else if (cpy_gray[i - 1][j] > cpy_gray[i][j - 1]) {
						labelSet[cpy_gray[i - 1][j]] = cpy_gray[i][j - 1];
						cpy_gray[i][j] = cpy_gray[i][j - 1];
					}
					else {
						cpy_gray[i][j] = cpy_gray[i - 1][j];
					}
				}
			}
		}
	}
	//第二次扫描
	for (int q = 1; q <= label; q++)
	{
		m = q;
		while (labelSet[m] != m)
		{
			m = labelSet[m];
			labelSet[q] = m;
		}
	}
	for (i = 1; i <= 119; i++)
	{
		for (j = 1; j <= 159; j++)
		{
			if (cpy_gray[i][j] != 0)
			{
				cpy_gray[i][j] = labelSet[cpy_gray[i][j]];
				S[cpy_gray[i][j]]++;
			}
		}
	}
	for (p = 0; p <= label; p++)
	{
		if (S[p] > max)
		{
			max = S[p];
			index = p;
		}
	}
	for (i = 0; i < 120; i++)
	{
		for (j = 0; j < 160; j++)
		{
			if (cpy_gray[i][j] != index)
			{
				cpy_gray[i][j] = 0;
			}
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       check_line()
// @param       void
// @return      void
// @function    检查线
//-------------------------------------------------------------------------------------------------------------------
void check_line()
{
	int i;
	for (i = High-1; i >=2; i--)
	{
		if (R_black[i] < L_black[i] + 10)
		{
			R_black[i] = 159;
			L_black[i] = 0;
		}
		M_black[i] = (R_black[i] + L_black[i]) / 2;
		//M_black[i] = 0.7 * M_black[i] + 0.3 * M_black[i - 1];
	}
	for (i = 1; i < High - 1; i++)
	{
		if (my_abs(M_black[i], M_black[i - 1]) > 8 && my_abs(M_black[i], M_black[i + 1]) > 8)M_black[i] = (M_black[i + 1] + M_black[i - 1]) / 2;
	}
	uint8 end_list = 0;
	if (L_black[scan_endline + 5 ]<10)end_list = 0;
	else if (R_black[scan_endline + 5] > Width - 10)end_list = Width - 1;
	for (i = scan_endline + 4; scan_endline >= 10 && i >= 5; i--)
	{
		M_black[i] = end_list;
	}
	printf("end:    %d   %d\n", scan_endline, end_list);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_spot_check()
// @param       void
// @return      void
// @function    环岛拐点检查
//-------------------------------------------------------------------------------------------------------------------
void circle_spot_check()
{
	if (circle_down_spot&& circle_mid_spot&&circle_down_spot < circle_mid_spot)
	{
		circle_down_spot = 0;
		circle_mid_spot = 0;
	}
	if (circle_mid_spot&& circle_up_spot&&circle_mid_spot < circle_up_spot)
	{
		circle_up_spot = 0;
		circle_mid_spot = 0;
	}
	if (circle_down_spot && circle_up_spot && circle_down_spot < circle_up_spot)
	{
		circle_down_spot = 0;
		circle_up_spot = 0;
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       zera_judge()
// @param       void
// @return      存在，不存在
// @function    斑马线判断
//-------------------------------------------------------------------------------------------------------------------
bool zera_judge()
{
	uint8 up = 0, down = 0;
	int mid=0;
	uint8 i;
	uint8 start=0, end=0;
	uint8 black_s[20];
	uint8 black_e[20];
	bool judge = 0;
	uint8 cnt1 = 0,cnt2=0;
	for (i = High - 5; i >= 10; i--)
	{
		if (!down)
		{
			if (L_black[i]>10&&R_black[i]<Width-10&&L_black[i+1]>10&&R_black[i+1]<Width-10 && road_width[i] > road_width[i - 1] + 3 && road_width[i + 1] > road_width[i - 1] + 5 && road_width[i + 2] > road_width[i - 1] + 8
				)
			{
				down = i;
			}
			if (down)
			{
				if (my_abs(L_black[i + 1], L_black[i + 2]) > 5 || my_abs(R_black[i + 2], R_black[i + 3]) > 5)down = 0;
			}
		}
	}
	for (i = 10; i < High -5; i++)
	{
		if (!up)
		{
			if (L_black[i] > 10 && R_black[i] < Width - 10 && L_black[i - 1]>10 && R_black[i - 1]<Width - 10 &&(int)(road_width[i]>road_width[i+1]) + 3 && road_width[i - 1] > road_width[i + 1] + 5 && road_width[i - 2] > road_width[i + 1] + 8
				)
			{
				up = i;
			}
			if (up)
			{
				if (my_abs(L_black[i -1], L_black[i - 2]) > 5 || my_abs(R_black[i - 2], R_black[i - 3]) > 5)up = 0;
			}
		}
	}
	if (up >= down + 30)down = 0;
	if (1)
	{
		if(up&& down)mid = (up + down) / 2;
		else if(up) mid = (up + 119) / 2;
		for (i = 3; i < Width/2; i++)
		{
			if (gray[mid][i - 2] == 0 && gray[mid][i - 1] == 0 && gray[mid][i])
			{
				start = i;
				break;
			}
		}
		for (i = Width-3; i >= Width/2; i--)
		{
			if (gray[mid][i+2] == 0 && gray[mid][i + 1] == 0 && gray[mid][i])
			{
				end = i;
				break;
			}
		}
		if (start && end)
		{
			for (i = start; i < end; i++)
			{
				if (gray[mid][i + 1] == 0 && gray[mid][i])
				{
					black_s[cnt1] = i;
					cnt1++;
					if (cnt1 >= 19 || cnt2 >= 19)break;
				}
				if (gray[mid][i] == 0 && gray[mid][i])
				{
					black_e[cnt2] = i;
					cnt2++;
					if (cnt1 >= 19 || cnt2 >= 19)break;
				}
			}
		}
	}
	uint8 min;
	min = cnt1 > cnt2 ? cnt1 : cnt2;
	printf("zera:::::::::::::: %d     %d    %d     %d   %d\n", up, down, start, end, min);
	if (min >= 6)
	{
		/*if (up && down)
		{
			advanced_regression(left, down + 8, down+3, up-3, up - 8);
			draw_line(left, down + 3, up - 3, parameterB, parameterA);
			advanced_regression(right, down + 8, down+3, up-3, up - 8);
			draw_line(right, down + 3, up - 3, parameterB, parameterA);
		}
		else*/ if (up)
		{
			regression(left, up - 5, up - 10);
			draw_line(left, up+30, up - 3, parameterB, parameterA);
			regression(right, up - 5, up - 10);
			draw_line(right, up+30, up - 3, parameterB, parameterA);
		}
		printf("zera   up:%d   down:%d\n", up, down);
		if(up&&down&&down>up+10)judge = 1;
	}
	else judge = 0;
	if (judge)
	{
		if (up >= 90 && start == 2)start = 1;
		
	}
	/*if (mid <= High - zera_position)judge=0;*/
	return judge;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       barrier_judge()
// @param       void
// @return      void
// @function    路障判断
//-------------------------------------------------------------------------------------------------------------------
void  barrier_judge()
{
	barrier = 0;
	barrier_down = 0; barrier_up = 0;
	static uint8 ps_protect = 0;
	uint8 tempi;
	uint16 line_s;
	uint8 mid;
	uint8 start=0, end=0;
	uint8 black_s[20], black_e[20], cnt1 = 0,cnt2=0;
	uint8 i, j;
	for (int i = High - start_line; i >= 20; i--)
	{
		if (road_width[i]<10 || road_width[i-5]<10||road_width[i+5]<10||L_black[i-2]<5||R_black[i-2]>Width-5)continue;
		if (!barrier_down)
		{
			if (road_width[i + 1] >10 &&road_width[i+2]>10 && road_width[i - 1] + 3< road_width[i] && my_abs(road_width[i - 1], road_width[i - 2])<6 && my_abs(road_width[i - 2], road_width[i - 3]) < 6 && my_abs(road_width[i - 3], road_width[i - 4]) < 6)
			{
				for (tempi = i; tempi >= i - 5; tempi--)
				{
					if (my_abs(road_width[tempi - 1], road_width[tempi - 2]) < 5 && my_abs(road_width[tempi - 2], road_width[tempi - 3]) < 5 && my_abs(road_width[tempi - 3], road_width[tempi - 4]) < 5)
					{
						if(road_width[tempi]+10<road_width[i])
						{
							barrier_down = tempi;
							if (R_black[barrier_down]+8 < R_black[i+2]&&my_abs(L_black[barrier_down+2], L_black[barrier_down + 3])<3 && my_abs(L_black[barrier_down + 3], L_black[barrier_down + 4] < 3) && my_abs(L_black[barrier_down + 4], L_black[barrier_down + 5] < 3))barrier_type = right;
							if (L_black[barrier_down] > L_black[i+2]+8 && my_abs(R_black[barrier_down + 2], R_black[barrier_down + 3]) < 3 && my_abs(R_black[barrier_down + 3], R_black[barrier_down + 4] < 3) && my_abs(R_black[barrier_down + 4], R_black[barrier_down + 5] < 3))barrier_type = left;
							break;
						}
					}
				}
			}
		}
		if (!barrier_up)
		{
			if (road_width[i - 1] > 5 && road_width[i - 2] > 5&&road_width[i]+3 < road_width[i-1]&&road_width[i]<road_width[i-2]&&road_width[i]<road_width[i-3]
				&&my_abs(road_width[i-1],road_width[i-2])<5&&my_abs(road_width[i-2],road_width[i-3])<5&&my_abs(road_width[i-3],road_width[i-4])<5)
			{
				if (road_width[barrier_up+2]<Width/3 && my_abs(road_width[i + 1], road_width[i + 2]) < 3 && my_abs(road_width[i + 2], road_width[i + 3]) < 3 && my_abs(road_width[i + 3], road_width[i + 4]) < 3)
				{
					barrier_up = i;
				}
			}
		}
		if (barrier_up && barrier_down&&barrier_down>barrier_up)break;
	}
	if (barrier_up && barrier_down && barrier_down >= barrier_up + 10)
	{
		uint8 up = barrier_up;
		uint8 down = barrier_down;
		if (1)
		{
			if (up && down)mid = (up + down) / 2;
			else if (up) mid = (up + 119) / 2;
			for (i = 3; i < Width / 2; i++)
			{
				if (gray[mid][i - 2] == 0 && gray[mid][i - 1] == 0 && gray[mid][i])
				{
					start = i;
					break;
				}
			}
			for (i = Width - 3; i >= Width / 2; i--)
			{
				if (gray[mid][i + 2] == 0 && gray[mid][i + 1] == 0 && gray[mid][i])
				{
					end = i;
					break;
				}
			}
			if (start && end)
			{
				for (i = start; i < end; i++)
				{
					if (gray[mid][i + 1] == 0 && gray[mid][i])
					{
						black_s[cnt1] = i;
						cnt1++;
						if (cnt1 >= 19 || cnt2 >= 19)break;
					}
					if (gray[mid][i] == 0 && gray[mid][i])
					{
						black_e[cnt2] = i;
						cnt2++;
						if (cnt1 >= 19 || cnt2 >= 19)break;
					}
				}
			}
		}
		if(my_abs(barrier_up, barrier_down)<70&& cnt1<5&& cnt2<5)
		{
			
			if (barrier_type)ps_protect++;
			if (ps_protect >= 3)
			{
				barrier = 1;
				ps_protect = 0;
			}
		}
		else if(cnt1>=6&& cnt2>=6)
		{
			if (up)
			{
				regression(left, up - 5, up - 10);
				draw_line(left, up + 30, up - 3, parameterB, parameterA);
				regression(right, up - 5, up - 10);
				draw_line(right, up + 30, up - 3, parameterB, parameterA);
			}
		}
	}
	printf("barrier  up:%d   down:%d   type:%d  \n", barrier_up, barrier_down, barrier_type);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       barrier_judge()
// @param       void
// @return      void
// @function    路障判断
//-------------------------------------------------------------------------------------------------------------------
void  barrier_search()
{
	barrier_down = 0; barrier_up = 0;
	static uint8 ps_protect = 100;
	int i = 0;
	uint8 tempi;
	for (i = High - start_line; i >= 30; i--)
	{
		if (road_width[i] < 10 || road_width[i - 5] < 10 || road_width[i + 5] < 10 || L_black[i]<5 || R_black[i]>Width - 5)continue;
		if (!barrier_up)
		{
			if (road_width[i] + 3 < road_width[i - 1] && road_width[i] < road_width[i - 2] && road_width[i] < road_width[i - 3]
				&& my_abs(road_width[i - 1], road_width[i - 2]) < 5 && my_abs(road_width[i - 2], road_width[i - 3]) < 5 && my_abs(road_width[i - 3], road_width[i - 4]) < 5)
			{
				if (my_abs(road_width[i + 1], road_width[i + 2]) < 3 && my_abs(road_width[i + 2], road_width[i + 3]) < 3 && my_abs(road_width[i + 3], road_width[i + 4]) < 3)
				{
					barrier_up = i;
				}
			}
		}
		if (!barrier_down)
		{
			if (road_width[i - 1] + 5 < road_width[i] && my_abs(road_width[i - 1], road_width[i - 2]) < 6 && my_abs(road_width[i - 2], road_width[i - 3]) < 6 && my_abs(road_width[i - 3], road_width[i - 4]) < 6)
			{
				for (tempi = i; tempi >= i - 5; tempi--)
				{
					if (my_abs(road_width[tempi - 1], road_width[tempi - 2]) < 5 && my_abs(road_width[tempi - 2], road_width[tempi - 3]) < 5 && my_abs(road_width[tempi - 3], road_width[tempi - 4]) < 5)
					{
						if (road_width[tempi] + 10 < road_width[i])
						{
							barrier_down = tempi;
							/*if (R_black[barrier_down] + 8 < R_black[i + 2])barrier_type = right;
							if (L_black[barrier_down] > L_black[i + 2] + 8)barrier_type = left;*/
							break;
						}
					}
				}
			}
		}
		if (barrier_up && barrier_down && barrier_down > barrier_up)break;
	}
	if (barrier_down && barrier_up)
	{
		/*regression(barrier_type, barrier_down + 8, barrier_down+3);
		draw_line(barrier_type, High - 1, barrier_up - 8, parameterB, parameterA);
		for (i = High - 1; i >= barrier_up; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
		printf("%d  %d\n", barrier_down, barrier_up);*/
	}
	if (barrier_down)
	{
		if (barrier_type == left)
		{
			two_point_regression(barrier_down - 3, L_black[barrier_down - 3], 0, Width-5);
			draw_two_point(1,barrier_down - 3,0, parameterk, parameterb);
			two_point_regression(barrier_down - 3, R_black[barrier_down - 3], 0, Width-5);
			draw_two_point(2, barrier_down+3, 0, parameterk, parameterb);
		}
		for (i = 0; i < High; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
	}
	else if (barrier_up)
	{
		for (i = High-1; i >=barrier_up; i--)
		{
			if (barrier_type == right)
			{
				if (M_black[i] >= 30)M_black[i] -= 30;
				else M_black[i] = 0;
			}
			else if (barrier_type == left)
			{
				if (M_black[i] <= Width - 31)M_black[i] += 30;
				else M_black[i] = Width - 1;
			}
		}
	}
	else
	{
		ps_protect--;
		if (ps_protect <= 1)
		{
			ps_protect = 100;
			barrier = 0;
		}
		for (i = High - 1; i >= 30; i--)
		{
			if (barrier_type == right)
			{
				if (M_black[i] >= 30)M_black[i] -= 30;
				else M_black[i] = 0;
			}
			else if (barrier_type == left)
			{
				if (M_black[i] <= Width - 31)M_black[i] += 30;
				else M_black[i] = Width - 1;
			}
		}
	}
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_mid_find(uint8 circle_type)
// @param       type
// @return      void
// @function    环岛-中拐点查找
//-------------------------------------------------------------------------------------------------------------------
void circle_mid_find(uint8 circle_type)
{
	int i;
	uint8 down = 0;
	if (circle_type == left)
	{
		for (i = High - 10; i >= 20; i--)
		{
			if (L_black[i] < 5 && L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i + 3] < 5)
			{
				if (L_black[i - 1] > L_black[i] && L_black[i - 2] > L_black[i - 1] && L_black[i - 3] > L_black[i - 2])
				{
					down = i;
					break;
				}
			}
		}
	}
	if (circle_type == right)
	{
		for (i = High - 10; i >= 20; i--)
		{
			if (R_black[i] > Width - 5 && R_black[i + 1] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5)
			{
				if (R_black[i - 1] < R_black[i] && R_black[i - 2] < R_black[i - 1] && R_black[i - 3] < R_black[i - 2])
				{
					down = i;
					break;
				}
			}
		}
	}
	circle_mid_spot = 0;
	if (circle_type == left)
	{
		for (i = down; i >= 20; i--)
		{
			if (!road_width[i])continue;
			if (!circle_mid_spot)
			{
				if (L_black[i] > L_black[i - 7] && L_black[i] > L_black[i + 7] && my_abs(L_black[i], L_black[i - 1]) < 10 && my_abs(L_black[i + 1], L_black[i]) < 10
					&& L_black[i] >= 10 && L_black[i - 1] >= 10 && L_black[i + 1] >= 10)
				{
					circle_mid_spot = i;
					break;
				}
			}
		}
	}
	else if (circle_type == right)
	{
		for (i = down; i >= 20; i--)
		{
			if (!road_width[i])continue;
			if (!circle_mid_spot)
			{
				if (R_black[i] < R_black[i - 7] && R_black[i] < R_black[i + 7] && my_abs(R_black[i], R_black[i - 1]) < 10 && my_abs(R_black[i + 1], R_black[i]) < 10
					&& R_black[i] <= Width - 10 && R_black[i - 1] <= Width - 10 && R_black[i + 1] <= Width - 10)
				{
					circle_mid_spot = i;
					break;
				}
			}
		}
	}
	printf("circle_mid_spot:    %d   down:     %d\n", circle_mid_spot,down);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_down_find(uint8 circle_type)
// @param       type
// @return      void
// @function    环岛-下拐点查找
//-------------------------------------------------------------------------------------------------------------------
void circle_down_find(uint8 circle_type)
{
	circle_down_spot = 0;
	int i;
	uint8 tempi;
	if (circle_type == left)
	{
		for (i = High - start_line; i >= 10; i--)
		{
			if (L_black[i] > 5 && L_black[i - 1] < 5 && L_black[i - 2] < 5 && L_black[i - 3] < 5 && L_black[i - 4] < 5)
			{
				for (tempi = i; tempi < i + 5; tempi++)
				{
					if (L_black[tempi] > 10 && my_abs(L_black[tempi], L_black[tempi + 1]) < 3 && my_abs(L_black[tempi + 1], L_black[tempi + 2]) < 3 && my_abs(L_black[tempi + 2], L_black[tempi + 3]) < 3
						&&L_black[tempi+1]>=L_black[tempi+3]&&L_black[tempi+2]>=L_black[tempi+4]&&road_width[tempi+2]>60)
					{
						circle_down_spot = tempi;
						break;
					}
				}
			}
			if (circle_down_spot)break;
		}
		if (L_black[circle_down_spot] < L_black[circle_down_spot-4] + 10)circle_down_spot = 0;
	}
	else if (circle_type == right)
	{
		for (i = High - start_line; i >= 10; i--)
		{
			if (R_black[i] <Width - 5 && R_black[i - 1] >Width - 5 && R_black[i - 2] > Width - 5 && R_black[i - 3] > Width - 5 && R_black[i - 4] > Width - 5)
			{
				for (tempi = i; tempi < i + 5; tempi++)
				{
					if (R_black[tempi] < Width - 10 && my_abs(R_black[tempi], R_black[tempi + 1]) < 3 && my_abs(R_black[tempi + 1], R_black[tempi + 2]) < 3 && my_abs(R_black[tempi + 2], R_black[tempi + 3]) < 3
						&&R_black[tempi+1]<=R_black[tempi+3]&&R_black[tempi+2]<=R_black[tempi+4]&&road_width[tempi+2]>60)
					{
						circle_down_spot = tempi;
						break;
					}
				}
			}
			if (circle_down_spot)break;
		}
		if (R_black[circle_down_spot]+10 > R_black[circle_down_spot-4])circle_down_spot = 0;
	}
	printf("down:   %d\n", circle_down_spot);
}
bool eight_area_find_cross_spot(uint8 type)//简易八领域寻找拐点
{
	sideways_type = 0;
	int i = 0, j = 0;
	uint8 temp_ld_cross = 0, temp_rd_cross = 0, temp_lu_cross = 0, temp_ru_cross = 0;
	bool ld_find = 0, rd_find = 0, lu_find = 0, ru_find = 0;
	uint8 tempi = 0;
	bool judge = 0;
	if (type == 1)
	{//左下拐点
		for (i = High - start_line; i >= 20; i--)
		{
			cross_left_down_line = 0; cross_left_down_flag = 0;
			if (my_abs(L_black[i], L_black[i + 1]) < 3 && my_abs(L_black[i + 2], L_black[i + 1]) < 3 && my_abs(L_black[i + 3], L_black[i + 2]) < 3 && L_black[i] > L_black[i - 1] + 5)
			{
				cross_left_down_line = i;
				temp_ld_cross = i;
				break;
			}
			else if (my_abs(L_black[i], L_black[i + 1]) < 3 && my_abs(L_black[i + 1], L_black[i + 2]) < 3 && my_abs(L_black[i + 2], L_black[i + 3]) < 5
				&& L_black[i] > L_black[i - 5] && L_black[i] > L_black[i + 5])
			{
				if (L_black[i] >= L_black[i - 5] + 5)
				{
					sideways_type = left;
					cross_left_down_line = i;
					temp_ld_cross = i;
					break;
				}
			}
		}
		
		if (temp_ld_cross)
		{
			for (i = L_black[temp_ld_cross] - 2; i >= 0; i--)
			{
				ld_find = 0;
				for (j = temp_ld_cross - 4; j <= temp_ld_cross + 4; j++)
				{
					if (gray[j][i] && gray[j - 1][i] && !gray[j + 1][i])
					{
						ld_find = 1;
						if (temp_ld_cross >= 3 && temp_ld_cross <= High - 5)temp_ld_cross = j;
						break;
					}
				}
				if (!ld_find)break;
			}
		}
		if (i <= 1 || L_black[cross_left_down_line] > i + 30)
		{
			cross_left_down_flag = 1;
			judge = 1;
		}
		if (cross_left_down_flag)if (my_abs(L_black[cross_left_down_line], L_black[cross_left_down_line - 3]) > 90 || my_abs(L_black[cross_left_down_line], L_black[cross_left_down_line + 3]) > 90) { cross_left_down_flag = 0; judge = 0; }
		if (cross_left_down_flag == 1)printf("ld:    %d\n", cross_left_down_line);
		if (cross_flag && cross_left_down_flag)last_left_down = cross_left_down_line;
		else if (!cross_flag)last_left_down = 0;
	}
	if (type == 2)
	{//右下拐点
		cross_right_down_flag = 0; cross_right_down_line = 0;
		for (i = High - start_line; i >= 20; i--)
		{
			if (my_abs(R_black[i], R_black[i + 1]) < 3 && my_abs(R_black[i + 2], R_black[i + 1]) < 3 && my_abs(R_black[i + 3], R_black[i + 2]) < 3 && R_black[i] + 5 < R_black[i - 1])
			{
				cross_right_down_line = i;
				temp_rd_cross = i;
				break;
			}
			else if (my_abs(R_black[i], R_black[i + 1]) < 3&& my_abs(R_black[i + 1], R_black[i + 2]) < 3 && my_abs(R_black[i + 2], R_black[i + 3]) < 3
				&& R_black[i] < R_black[i - 5] && R_black[i] < R_black[i + 5])
			{
				if (R_black[i] + 5 <= R_black[i - 5])
				{
					sideways_type = right;
					cross_right_down_line = i;
					temp_rd_cross = i;
					break;
				}
			}
		}
		
		if (temp_rd_cross)
		{
			for (i = R_black[temp_rd_cross] + 2; i <= Width - 1; i++)
			{
				rd_find = 0;
				for (j = temp_rd_cross - 4; j <= temp_rd_cross + 4; j++)
				{
					if (gray[j][i] && gray[j - 1][i] && !gray[j + 1][i])
					{
						rd_find = 1;
						if (temp_rd_cross >= 3 && temp_rd_cross <= High - 5)temp_rd_cross = j;
						break;
					}
				}
				if (!rd_find)break;
			}
		}
		if (i >= Width - 1 || R_black[cross_right_down_line] + 30 < i)
		{
			cross_right_down_flag = 1;
			judge = 1;
		}
		if (cross_right_down_flag)if (my_abs(R_black[cross_right_down_line], R_black[cross_right_down_line - 3]) >90 || my_abs(R_black[cross_right_down_line], R_black[cross_right_down_line + 3])>90)cross_right_down_flag = 0;
		if (cross_right_down_flag == 1)printf("rd:    %d\n", cross_right_down_line);
		if (cross_flag && cross_right_down_flag)last_right_down = cross_right_down_line;
		else if (!cross_flag)last_right_down = 0;
	}
	//左上拐点
	if (type == 3)
	{
		cross_left_up_flag = 0; cross_left_up_line = 0;
		for (i = High - start_line; i >= 30; i--)
		{
			if (L_black[i + 3] < 5 && L_black[i + 2] < 5 && L_black[i + 1] < 5)
			{
				for (tempi = i; tempi >= i - 5; tempi--)
				{
					if (L_black[tempi] < Width - 5 && L_black[tempi]>10 && L_black[tempi - 1] > 5 && my_abs(L_black[tempi], L_black[tempi - 1]) < 3 && my_abs(L_black[tempi - 1], L_black[tempi - 2]) < 3 && my_abs(L_black[tempi - 2], L_black[tempi - 3]) < 3)
					{
						cross_left_up_line = (byte)(tempi);
						cross_left_up_flag = 1;
						break;
					}
				}
				if (cross_left_up_flag == 1)
				{
					if (L_black[cross_left_up_line] < 5 + L_black[cross_left_up_line + 3]||my_abs(L_black[cross_left_up_line], L_black[cross_left_up_line+3])>Width/2)
					{
						cross_left_up_flag = 0;
					}
				}
			}
		}
		if (cross_left_up_flag)if (my_abs(L_black[cross_left_up_line], L_black[cross_left_up_line - 3]) > 90 || my_abs(L_black[cross_left_up_line], L_black[cross_left_up_line + 3])>90)cross_left_up_flag = 0;
		if (cross_left_up_flag == 1)
		{
			printf("lu:    %d\n", cross_left_up_line);
			judge = 1;
		}
		if (cross_flag && cross_left_up_flag)last_left_up = cross_left_up_line;
		else if (!cross_flag)last_left_up = 0;
	}
	//右上拐点
	if (type == 4)
	{
		cross_right_up_flag = 0; cross_right_up_line = 0;
		for (i = High - start_line; i >=30; i--)
		{
			if (R_black[i + 3] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 1] > Width - 5)
			{
				for (tempi = i; tempi >= i - 5; tempi--)
				{
					if (R_black[tempi] > 15 && R_black[tempi - 1] < Width - 5 && my_abs(R_black[tempi], R_black[tempi - 1]) < 3 && my_abs(R_black[tempi - 1], R_black[tempi - 2]) < 3 && my_abs(R_black[tempi - 2], R_black[tempi - 3]) < 3)
					{
						cross_right_up_line = (byte)(tempi);
						cross_right_up_flag = 1;
						break;
					}
				}
				if (cross_right_up_flag == 1)
				{
					if (R_black[cross_right_up_line + 3] < 5 + R_black[cross_right_up_line]||my_abs(R_black[cross_right_up_line], R_black[cross_right_up_line + 3])>Width/2)
					{
						cross_right_up_flag = 0;
					}
				}
			}
		}
		if (cross_right_up_flag)if (my_abs(L_black[cross_right_up_line], L_black[cross_right_up_line - 3]) > 90 || my_abs(L_black[cross_right_up_line], L_black[cross_right_up_line + 3])>90)cross_right_up_flag = 0;
		if (cross_right_up_flag == 1)
		{
			printf("ru:    %d\n", cross_right_up_line);
			judge = 1;
		}
		if (cross_flag && cross_right_up_flag)last_right_up = cross_right_up_line;
		else if (!cross_flag)last_right_up = 0;
	}
	return judge;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_up_find(uint8 circle_type)
// @param       type
// @return      void
// @function    环岛-上拐点查找
//-------------------------------------------------------------------------------------------------------------------
void circle_up_find(uint8 circle_type)
{
	circle_up_spot = 0;
	circle_fork = 0;
	int i;
	if (circle_type == left)
	{
		for (i = High - start_line; i >= 5; i--)
		{
			if (!road_width[i])continue;
			if (L_black[i]>L_black[i+1]+5&&road_width[i-1]>5
				&& L_black[i]>5 && my_abs(L_black[i], L_black[i - 1]) < 3 && my_abs(L_black[i - 1], L_black[i - 2]) < 3 && my_abs(L_black[i - 2], L_black[i - 3]) < 3)
			{
				circle_up_spot = i;
				circle_fork = 1;
				break;
			}
			else if (R_black[i] + 10 < R_black[i + 1] && R_black[i - 1] + 10 < R_black[i + 1] && R_black[i - 2] + 10 < R_black[i + 1]
				&& my_abs(R_black[i + 1], R_black[i + 2]) < 3 && my_abs(R_black[i + 2], R_black[i + 3]) < 3 && my_abs(R_black[i + 3], R_black[i + 4]) < 3
				&& my_abs(R_black[i], R_black[i - 1]) < 10)
			{
				circle_up_spot = i;
				break;
			}
		}
	}
	else if (circle_type == right)
	{
		for (i = High - start_line; i >= 5; i--)
		{
			if (!road_width[i])continue;
			if (R_black[i]+10<R_black[i+1]&&road_width[i-1]>5
				&& R_black[i] < Width - 10 && my_abs(R_black[i], R_black[i - 1]) < 3 && my_abs(R_black[i - 1], R_black[i - 2]) < 3 && my_abs(R_black[i - 2], R_black[i - 3]) < 3)
			{
				circle_up_spot = i;
				circle_fork = 1;
				break;
			}
			else if (L_black[i] > L_black[i + 1] + 10 && L_black[i - 1] > L_black[i + 1] + 10 && L_black[i - 2] > L_black[i + 1] + 10
				&& my_abs(L_black[i + 1], L_black[i + 2]) < 3 && my_abs(L_black[i + 2], L_black[i + 3]) < 3 && my_abs(L_black[i + 3], L_black[i + 4]) < 3
				&& my_abs(L_black[i], L_black[i - 1]) < 10)
			{
				circle_up_spot = i;
				break;
			}
		}
	}
	printf("up:   %d     style:   %d\n", circle_up_spot,circle_fork);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_state_judge(uint8 circle_type)
// @param       type
// @return      void
// @function    环岛-出环状态判断
//-------------------------------------------------------------------------------------------------------------------
bool circle_state_judge(uint8 circle_type)
{
	uint8 sum = 0;
	bool judge = 0;
	int i;
	for (i = High - start_line; i >= 40; i--)
	{
		if (!road_width[i])continue;
		if (road_width[i] > Width - 5)
		{
			sum++;
			if (sum >= High / 3)
			{
				judge = 1;
				break;
			}
		}
		else sum = 0;
	}
	printf("sum:  %d\n", sum);
	return judge;
}
//-------------------------------------------------------------------------------------------------------------------
// @brief       circle_find_leave_spot(uint8 circle_type)
// @param       uint8 circle_type 环岛方向 左右
// @return      void
// @function    环岛寻找出环拐点
//-------------------------------------------------------------------------------------------------------------------
void circle_find_leave_spot(uint8 circle_type)
{
	uint8 i;
	circle_leave_spot = 0;
	uint8 tempi;
	if (circle_type == left)
	{
		for (i = High - start_line; i >= 30; i--)
		{
			if (!road_width[i])continue;
			if (R_black[i] < R_black[i - 3] &&R_black[i]<R_black[i+3]&&R_black[i]<R_black[i-5]&&R_black[i]<R_black[i+5]&&R_black[i-3]<R_black[i-5]&&R_black[i+3]<R_black[i+5]
				&& my_abs(R_black[i], R_black[i + 1]) < 3 && my_abs(R_black[i + 1], R_black[i + 2]) < 3 && my_abs(R_black[i + 2], R_black[i + 3]) < 3
				&&R_black[i]<Width-8)
			{
				circle_leave_spot = i;
				break;
			}
			else if (R_black[i]>Width-5&&R_black[i-1]>Width-5&&R_black[i-2]>Width-5&&R_black[i-3]>Width - 5&&R_black[i+1]<Width-8)
			{
				for (tempi = i; tempi < i + 4; tempi++)
				{
					if (R_black[tempi] < Width - 8 && my_abs(R_black[tempi], R_black[tempi + 1]) < 3 && my_abs(R_black[tempi + 1],R_black[tempi + 2]) < 3&&my_abs(R_black[tempi+2],R_black[tempi+3])<3)
					{
						circle_leave_spot = i;
						break;
					}
				}
			}
			if (circle_leave_spot)break;
		}
	}
	else if (circle_type == right)
	{
		for (i = High - start_line; i >= 30; i--)
		{
			if (!road_width[i])continue;
			if (L_black[i] > L_black[i - 3] && L_black[i] > L_black[i + 3] && L_black[i] > L_black[i - 5] && L_black[i] > L_black[i + 5] && L_black[i - 3] > L_black[i - 5] && L_black[i + 3] > L_black[i + 5]
				&& my_abs(L_black[i], L_black[i + 1]) < 3 && my_abs(L_black[i + 1], L_black[i + 2]) < 3 && my_abs(L_black[i + 2], L_black[i + 3]) < 3
				&& L_black[i] >8)
			{
				circle_leave_spot = i;
				break;
			}
			else if (L_black[i] <  5 && L_black[i - 1] <  5 && L_black[i - 2] < 5 && L_black[i - 3] < 5 && L_black[i + 1] > 8)
			{
				for (tempi = i; tempi < i + 4; tempi++)
				{
					if (L_black[tempi] > 8 && my_abs(L_black[tempi], L_black[tempi + 1]) < 3 && my_abs(L_black[tempi + 1], L_black[tempi + 2]) < 3 && my_abs(L_black[tempi + 2], L_black[tempi + 3]) < 3)
					{
						circle_leave_spot = i;
						break;
					}
				}
			}
			if (circle_leave_spot)break;
		}
	}
	printf("leave:   %d\n", circle_leave_spot);
}
void L_R_nodown_judge(uint8 type)
{
	uint8 mid_up = 0, mid_down = 0, mid = 0;
	uint8 i;
	static uint8 protect = 0;
	circle_up_find(type);
	if (type == right)
	{
		for (i = 3; i < High - start_line; i++)
		{
			if (!mid_up && R_black[i] > Width - 5 && R_black[i - 1] > Width - 5 && R_black[i - 2] > Width - 5 && R_black[i - 3] > Width - 5 && R_black[i + 1] < R_black[i] && R_black[i + 2] < R_black[i + 1] && R_black[i + 3] < R_black[i + 2] && R_black[i + 3] < Width - 5)
			{
				mid_up = i;
			}
			if (!mid_down && R_black[i] > Width - 5 && R_black[i + 1] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5 && R_black[i - 1] < R_black[i] && R_black[i - 2] < R_black[i - 1] && R_black[i - 3] < R_black[i - 2]
				&& my_abs(R_black[i - 1], R_black[i - 2]) < 15 && my_abs(R_black[i], R_black[i - 1]) < 20 && my_abs(R_black[i - 3], R_black[i - 2]) < 15)
			{
				mid_down = i;
			}
		}
		for (i = High - start_line; i >= 4; i--)
		{
			if (R_black[i] < R_black[i - 7] && R_black[i] < R_black[i + 7] && R_black[i] < Width - 5)
			{
				mid = i;
				break;
			}
		}
		if (mid && mid_down && mid_up && mid_down > mid_up + 15 && mid >= 15)
		{
			if (my_abs(mid, (mid_down + mid_up) / 2) <= 15&&circle_up_spot&&circle_up_spot<mid+10)
			{
					circle_type = right;
					circle_state = 1;
	
			}
		}
		printf("%d    %d     %d\n", mid_up, mid, mid_down);
	}
	if (type == left)
	{
		for (i = 3; i < High - start_line; i++)
		{
			if (!mid_up && L_black[i] < 5 && L_black[i - 1] < 5 && L_black[i - 2] < 5 && L_black[i - 3] < 5 && L_black[i + 1] > L_black[i] && L_black[i + 2] > L_black[i + 1] && L_black[i + 3] > L_black[i + 2] && L_black[i + 3]>5)
			{
				mid_up = i;
			}
			if (!mid_down && L_black[i] < 5 && L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i + 3] < 5 && L_black[i - 1] > L_black[i] && L_black[i - 2] > L_black[i - 1] && L_black[i - 3] > L_black[i - 2]
				&& my_abs(L_black[i - 1], L_black[i - 2]) < 15 && my_abs(L_black[i], L_black[i - 1]) < 20 && my_abs(L_black[i - 3], L_black[i - 2]) < 15)
			{
				mid_down = i;
			}
		}
		for (i = High - start_line; i >= 10; i--)
		{
			if (L_black[i] > L_black[i - 7] && L_black[i] > L_black[i + 7] && L_black[i] > 5)
			{
				mid = i;
				break;
			}
		}
		if (mid && mid_down && mid_up && mid_down > mid_up + 15 && mid >= 15)
		{
			if (my_abs(mid, (mid_down + mid_up) / 2) <= 15 &&circle_up_spot&& circle_up_spot < mid + 10)
			{
					circle_type = left;
					circle_state = 1;
			}
			printf("second:%d    %d     %d!!!!\n", mid_up, mid, mid_down);
		}
	}
}
void check_big_circle(uint8 type)
{
	uint8 j;
	if (type == left)
	{
		for (j = L_black[circle_mid_spot]-2; j >= 2; j--)
		{
			if (gray[circle_mid_spot][j-1] && gray[circle_mid_spot][j] && !gray[circle_mid_spot][j + 1])
			{
				small_circle_flag = 1;
			}
		}
	}
	if (type == right)
	{
		for (j = R_black[circle_mid_spot] + 2; j <=Width- 2; j++)
		{
			if (!gray[circle_mid_spot][j-1] && gray[circle_mid_spot][j] && gray[circle_mid_spot][j + 1])
			{
				small_circle_flag = 2;
			}
		}
	}
}
void L_R_circle_judge()
{
	static uint8 protect=0;
	int ls = my_line_offset(1, High - 20, 40);
	int rs = my_line_offset(2, High -20, 40);
	int i;
	circle_type = 0;
	circle_down_spot = 0;
	circle_mid_spot = 0;
	circle_state = 0;
	int up = 0, down = 0;
	find_max_list();
	printf("ls:%d   rs:%d", ls, rs);
	if (ls >= 400 && rs <= 150 || left_losecountm >= right_losecountm + 20 && right_losecountm < 20)
	{
		circle_find_line();
		circle_down_find(left);
		find_cross_spot();
		if (circle_down_spot&&!cross_right_up_flag&&!cross_right_down_flag)
		{
			circle_mid_find(left);
			circle_up_find(left);
			if (circle_mid_spot && circle_down_spot > circle_mid_spot + 10)
			{
				protect++;
				if (protect >= 3)
				{
					circle_type = left;
					circle_state = 1;
					protect = 0;
				}
				regression(circle_type, High - 1, circle_down_spot + 3);
				draw_line(circle_type, circle_down_spot + 3, circle_mid_spot, parameterB, parameterA);
				for (i = circle_down_spot + 3; i >= circle_mid_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			printf("first judge\n");
		}
		else if (!cross_right_up_flag && !cross_right_down_flag)
		{
			L_R_nodown_judge(left);
			printf("second judge\n");
		}
		printf("circle_judge   down: %d   mid: %d   up: %d\n", circle_down_spot, circle_mid_spot, circle_up_spot);
	}
	else if (rs >=400 && ls <= 150 || right_losecountm >= left_losecountm + 20 && left_losecountm < 20)
	{
		circle_find_line();
		circle_down_find(right);
		find_cross_spot();
		if (circle_down_spot&&!cross_left_up_flag && !cross_left_down_flag)
		{
			circle_mid_find(right);
			circle_up_find(right);
			if (circle_mid_spot && circle_down_spot > circle_mid_spot + 10 )
			{
				protect++;
				if (protect >= 3)
				{
					circle_type = right;
					circle_state = 1;
					protect = 0;
				}
				regression(circle_type, High - 1, circle_down_spot + 3);
				draw_line(circle_type, circle_down_spot + 3, circle_mid_spot, parameterB, parameterA);
				for (i = circle_down_spot + 3; i >= circle_mid_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			printf("first judge\n");
		}
		else if (!cross_left_up_flag && !cross_left_down_flag)
		{
			L_R_nodown_judge(right);
			printf("second judge\n");
		}
		printf("circle_judge   down: %d   mid: %d   up: %d\n", circle_down_spot, circle_mid_spot, circle_up_spot);
	}
	if(circle_type)check_big_circle(circle_type);
	//printf("ls:%d     rs:%d\nleft_lose: %d  right_lose: %d\n", ls, rs, left_losecountm, right_losecountm);
}
//-------------------------------------------------------------------------------------------------------------------
// @brief      bool curve_judge()
// @param       void
// @return      是否为弯道
// @function    弯道判断判断
//-------------------------------------------------------------------------------------------------------------------
bool curve_judge()
{
	int i;
	uint8 break_line = 0;
	uint8 l_lose = 0, r_lose = 0;
	bool judge = 0;
	uint8 type = 0;
	uint8 first_type = 1;
	for (i = 20; i < High-10; i++)
	{
		if (my_abs(M_black[i], M_black[i - 2]) > 10)break_line = i;
	}
	uint8 curve_line = 0;
	if (break_line < 20)first_type = 0;
	if (1)
	{
		for (i =5; i <=High- 1; i++)
		{
			if (L_black[i] < 5)l_lose++;
			if (R_black[i] > Width - 5)r_lose++;
			//if(L_black[i] > 5&& L_black[i - 3] > 5 && L_black[i + 3] > 5 && (L_black[i] > L_black[i - 5] && L_black[i] > L_black[i + 5] || L_black[i] < L_black[i - 5] && L_black[i] < L_black[i + 5])
			//|| R_black[i] < Width - 5 && R_black[i - 3] < Width - 5 && R_black[i + 3] < Width - 5 && (R_black[i] > R_black[i - 5] && R_black[i] > R_black[i + 5] || R_black[i] < R_black[i - 5] && R_black[i] < R_black[i + 5]))curve_line=i;
		}
		if (l_lose > 20 && r_lose < 10)
		{
			if (my_line_offset(0, High - 1, break_line) >= 150)
			{
				judge = 1;
				type = right;
			}
		}
		if (r_lose > 20 && l_lose < 10)
		{
			if (my_line_offset(1, High - 1, break_line) >= 150)
			{
				judge = 1;
				type = left;
			}
		}
		
	}
	if (break_line && my_line_offset(1, High - 1, break_line) >= 300 && my_line_offset(2, High - 1, break_line) >= 300)judge = 1;
	else if (my_line_offset(1, 60, 20) >= 50 && my_line_offset(1, High - 1, 40) >= 50 && my_line_offset(1, High - 1, 0) >= 1500 || my_line_offset(2, 60, 20) >= 50 && my_line_offset(2, High - 1, 0) >= 1500 && my_line_offset(2, High - 1, 40) >= 50)judge = 1;
	else if (my_line_offset(1, High - 1, 0) >= 500 && my_line_offset(2, High - 1, 0) >= 500)judge = 1;
	else if (l_lose > High / 3 || r_lose > High / 3)judge = 1;
	if (judge)printf("curve    %d        %d          %d\n",break_line, my_line_offset(1, High - 1, break_line), my_line_offset(2, High - 1, break_line));
	printf("curve_judge    %d        %d       %d\n", break_line, my_line_offset(1, High - 1, 0), my_line_offset(2, High - 1, 0));
	printf("curve_line:   %d\n", curve_line);
	/*if (judge)
	{
		if (type == right)
		{
			two_point_regression(High-1, R_black[High-1],break_line*1.5, R_black[(uint8)(break_line * 1.5)]);
			draw_two_point(right, High - 1, break_line * 1.5, parameterk, parameterb);
			two_point_regression(break_line * 1.5, R_black[(uint8)(break_line * 1.5)], break_line,0);
			draw_two_point(right, break_line * 1.5, break_line, parameterk, parameterb);
		}
		else if (type == left)
		{
			two_point_regression(High - 1, L_black[High - 1], break_line * 1.5, L_black[(uint8)(break_line * 1.5)]);
			draw_two_point(left, High - 1, break_line * 1.5, parameterk, parameterb);
			two_point_regression(break_line * 1.5, L_black[(uint8)(break_line * 1.5)], break_line, Width);
			draw_two_point(left, break_line * 1.5, break_line, parameterk, parameterb);
		}
		for (i = High - 1; i >= break_line; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
		printf("judge :%d ls: %d  rs: %d\n",judge, l_lose, r_lose);
	}*/
	curve_flag = judge;
	return judge;
}
//1 左下，2右下，3左上，4右上
//-------------------------------------------------------------------------------------------------------------------
// @brief       card_find
// @param       void
// @return      void
// @function    判断赛道两旁是否有卡片
//-------------------------------------------------------------------------------------------------------------------
void average_filter()
{
	uint8 i, j,a,b;
	uint16 temp_B = 0,temp_R,temp_G;
	for (i = 1; i < High - 1; i++)
	{
		for (j = 1; j < Width - 1; j++)
		{
			temp_B = 0;
			temp_R = 0;
			temp_G = 0;
			for (a = i - 1; a <= i + 1; a++)
			{
				for (b = j - 1; b <= j + 1; b++)
				{
					temp_B += phB[a][b];
					temp_G += phG[a][b];
					temp_R += phR[a][b];
				}
			}
			phB[i][j] = temp_B / 9;
			phG[i][j] = temp_G / 9;
			phR[i][j] = temp_R / 9;
		}
	}
}
/*
* 总图象处理
*/
uint8 last_down[2];
bool check_flag = 1;
void circle_process()
{
	uint8 left_lose = 0, right_lose = 0;
	uint8 left_losem = 0, right_losem = 0;
	int i, j;
	static int circle_protect = 2;
	if (circle_state == 1)
	{
		circle_mid_find(circle_type);
		circle_down_find(circle_type);
		circle_up_find(circle_type);
		if (circle_down_spot&&!circle_mid_spot)
		{
			if (circle_type == left)
			{
				last_down[0] = circle_down_spot;
				last_down[1] = L_black[circle_down_spot];
			}
			else if (circle_type == right)
			{
				last_down[0] = circle_down_spot;
				last_down[1] = R_black[circle_down_spot];
			}
			regression(circle_type, circle_down_spot + 10, circle_down_spot + 3);
			draw_line(circle_type, circle_down_spot + 3, circle_mid_spot, parameterB, parameterA);
			for (i = circle_down_spot + 3; i >= 10; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			printf("only down\n");
		}
		else if(!last_down[1])
		{
			if (circle_type == left)
			{
				if (L_black[119] >= 10 && L_black[119] <= 50)
				{
					last_down[0] = 119;
					last_down[1] = L_black[119];
				}
				else
				{
					last_down[0] = High - 10;
					last_down[1] = 10;
				}
			}
			else if (circle_type == right)
			{
				if (R_black[119] <= 10 && R_black[119] <= Width-50)
				{
					last_down[0] = 119;
					last_down[1] = R_black[119];
				}
				else
				{
					last_down[0] = High - 10;
					last_down[1] = Width - 10;
				}
			}
		}
		if (circle_down_spot && circle_mid_spot && circle_down_spot > circle_mid_spot + 10&& circle_down_spot<90)
		{
			regression(circle_type, circle_down_spot + 10, circle_down_spot + 3);
			draw_line(circle_type, circle_down_spot + 3, circle_mid_spot, parameterB, parameterA);
			for (i = circle_down_spot + 3; i >= circle_mid_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			printf("all\n");
		}
		else if (circle_mid_spot)
		{
			if (circle_type == left&&last_down[1]>Width/2)
			{
				last_down[0] = High - 10;
				last_down[1] = 10;
			}
			else if (circle_type == right&&last_down[1]<Width/2)
			{
				last_down[0] = High - 10;
				last_down[1] = Width - 10;
			}
			if (circle_type == right)two_point_regression(last_down[0], last_down[1], circle_mid_spot, R_black[circle_mid_spot]);
			else if (circle_type == left)two_point_regression(last_down[0], last_down[1], circle_mid_spot, L_black[circle_mid_spot]);
			draw_two_point(circle_type, High - 1, circle_mid_spot, parameterk, parameterb);
			for (i = High - 1; i >= circle_mid_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			printf("only mid    \n");
		}
		else if (circle_up_spot&&circle_up_spot>=20)
		{
			regression(circle_type, circle_up_spot - 3, circle_up_spot - 8);
			draw_line(circle_type, High - 1, circle_up_spot, parameterB, parameterA);
			for (i = High - 1; i >= circle_up_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			circle_protect--;
			if (circle_protect <= 0)
			{
				circle_protect = 2;
				circle_state++;
			}
		}
	}
	else if (circle_state == 2)
	{
		circle_up_find(circle_type);
		circle_mid_find(circle_type);
		if (circle_up_spot && !circle_mid_spot)
		{
			if (circle_type == left)
			{
				if (circle_up_spot)
				{
					two_point_regression(circle_up_spot, R_black[circle_up_spot], High - 1, R_black[High - 1]);
					draw_two_point(right, High - 1, circle_up_spot, parameterk, parameterb);
					for (i = High - 1; i >= circle_up_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
					for (i = 0; i <= circle_up_spot; i++)
					{
						if (circle_type == left)M_black[i] = 0;
						else M_black[i] = Width - 1;
					}
					circle_protect--;
				}
				if (circle_protect <= 0)
				{
					circle_protect = 2;
					circle_state++;
				}
			}
			else if (circle_type == right)
			{
				if (circle_up_spot)
				{
					two_point_regression(circle_up_spot, L_black[circle_up_spot], High - 1, L_black[High - 1]);
					draw_two_point(left, High - 1, circle_up_spot, parameterk, parameterb);
					for (i = High - 1; i >= circle_up_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
					circle_protect--;
				}
				if (circle_protect <= 0)
				{
					circle_protect = 2;
					circle_state++;
				}
			}
		}
		else if(circle_up_spot&&circle_mid_spot&&circle_up_spot+10<circle_mid_spot)
		{
			if (circle_type == left)
			{
				two_point_regression(circle_mid_spot, L_black[circle_mid_spot],last_down[0], last_down[1]);
				draw_two_point(left, High - 1, 60, parameterk, parameterb);
				for (i = High - 1; i >= 60; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			else if (circle_type == right)
			{
				two_point_regression(circle_mid_spot, R_black[circle_mid_spot], last_down[0], last_down[1]);
				draw_two_point(left, High - 1,60, parameterk, parameterb);
				for (i = High - 1; i >= 60; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
		}
	}
	else if (circle_state == 3)
	{
		circle_up_find(circle_type);
		if (circle_type == left)
		{
			if (circle_up_spot)
			{
				if (!circle_fork)two_point_regression(circle_up_spot, R_black[circle_up_spot], High - 1, R_black[High - 1]);
				else
				{
					two_point_regression(circle_up_spot, L_black[circle_up_spot], High - 1, R_black[High - 1]);
				}
				draw_two_point(right, High - 1, circle_up_spot, parameterk, parameterb);
				uint8 end_line = 0;
				for (i = High - 6; i > 5; i--)
				{
					M_black[i] = (L_black[i] + R_black[i]) / 2;
					if (L_black[i] < 5 && L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i - 1]>5)
					{
						end_line = i;
						break;
					}
				}
				for(i= end_line;i>5;i--)M_black[i] = 0;
				if (end_line)check_flag = 0;
				else check_flag = 1;
			}
			else if (!circle_up_spot)
			{
				two_point_regression(High - 1, Width - 1, 40, Width / 2);
				draw_two_point(right, High - 1, 40, parameterk, parameterb);
				for (i = 40; i <= High - 1; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
				circle_protect--;
			}
			if (circle_protect <= 0)
			{
				circle_protect = 2;
				circle_state++;
			}
		}
		else if (circle_type == right)
		{
			if (circle_up_spot)
			{
				if (!circle_fork)two_point_regression(circle_up_spot, L_black[circle_up_spot], High - 1, L_black[High - 1]);
				else two_point_regression(circle_up_spot, R_black[circle_up_spot], High - 1, L_black[High - 1]);
				draw_two_point(left, High - 1, circle_up_spot, parameterk, parameterb);
				uint8 end_line = 0;
				for (i = High - 6; i > 5; i--)
				{
					M_black[i] = (L_black[i] + R_black[i]) / 2;
					if (R_black[i] >Width- 5 && R_black[i + 1] >Width- 5 && R_black[i + 2] >Width- 5 && R_black[i - 1]<Width-10)
					{
						end_line = i;
						break;
					}
				}
				for (i = end_line; i > 5; i--)M_black[i] = Width-1;
				if (end_line)check_flag = 0;
				else check_flag = 1;
			}
			else if (!circle_up_spot)
			{
				circle_protect--;
				two_point_regression(High - 1, 0, 40, Width / 2);
				draw_two_point(left, High - 1, 40, parameterk, parameterb);
				for (i = 40; i <= High - 1; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			if (circle_protect <= 0)
			{
				circle_protect = 2;
				circle_state++;
			}
		}
	}
	else if (circle_state == 4)
	{
		circle_find_leave_spot(circle_type);
		if (circle_leave_spot)
		{
			if (circle_type == left)
			{
				two_point_regression(40, 0, circle_leave_spot, R_black[circle_leave_spot + 2]);
				draw_two_point(right, circle_leave_spot + 2, 40, parameterk, parameterb);
				for (i = 40; i <= circle_leave_spot + 2; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
				for (i = 0; i < 40; i++)M_black[i] = 0;
			}
			else
			{
				two_point_regression(40, 159, circle_leave_spot, L_black[circle_leave_spot + 2]);
				draw_two_point(left, circle_leave_spot + 2, 40, parameterk, parameterb);
				for (i = 40; i <= circle_leave_spot + 2; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
				for (i = 0; i < 40; i++)M_black[i] = Width - 1;
			}
			circle_protect--;
			if (circle_protect <= 0)
			{
				circle_state++;
				circle_protect = 2;
			}
		}
		if (circle_leave_spot)check_flag = 0;
		else check_flag = 1;
	}
	else if (circle_state == 5)
	{
		find_cross_spot();
		check_flag = 1;
		circle_up_spot = 0;
		if (circle_type == left)
		{
			if (cross_left_up_flag && L_black[cross_left_up_line - 1] < Width / 2)
				circle_up_spot = cross_left_up_line;
		}
		else if (circle_type == right)
		{
			if (cross_right_up_flag && R_black[cross_right_up_line - 1] > Width / 2)
				circle_up_spot = cross_right_up_line;
		}
		circle_find_leave_spot(circle_type);
		if (circle_leave_spot)
		{
			if (circle_type == left)
			{
				two_point_regression(40, 0, circle_leave_spot, R_black[circle_leave_spot + 2]);
				draw_two_point(right, circle_leave_spot + 2, 40, parameterk, parameterb);
				for (i = 40; i <= circle_leave_spot + 2; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			else
			{
				two_point_regression(40, 159, circle_leave_spot, L_black[circle_leave_spot + 2]);
				draw_two_point(left, circle_leave_spot + 2, 40, parameterk, parameterb);
				for (i = 40; i <= circle_leave_spot + 2; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
		}
		else
		{
			if (circle_type == left)
			{
				for (i = High - 20; i >= 30; i--)
				{
					if (R_black[i] > Width - 5)right_lose++;
					else
					{
						if (right_lose > right_losem)right_losem = right_lose;
						right_lose = 0;
					}
				}
				if (right_losem > 20)
				{
					two_point_regression(High - 1, Width - 1, 40, Width / 3);
					draw_two_point(right, High - 1, 40, parameterk, parameterb);
					for (i = 40; i < High; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
				}
			}
			else
			{
				for (i = High - 20; i >= 30; i--)
				{
					if (L_black[i] < 5)left_lose++;
					else
					{
						if (left_lose > left_losem)left_losem = left_lose;
						left_lose = 0;
					}
				}
				if (left_losem > 20)
				{
					two_point_regression(High - 1, 0, 40, Width * 2 / 3);
					draw_two_point(left, High - 1, 40, parameterk, parameterb);
					for (i = 40; i < High; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
				}
			}
		}
		if (circle_up_spot)
		{
			regression(circle_type, circle_up_spot - 3, circle_up_spot - 8);
			draw_line(circle_type, High - 1, circle_up_spot, parameterB, parameterA);
			for (i = High - 1; i >= circle_up_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			circle_protect--;
			if (circle_protect <= 0)
			{
				circle_protect = 2;
				circle_state++;
			}
		}
	}
	else if (circle_state == 6)
	{
		find_cross_spot();
		circle_up_spot = 0;
		if (circle_type == left)
		{
			if (cross_left_up_flag && L_black[cross_left_up_line - 1] < Width / 2)
				circle_up_spot = cross_left_up_line;
		}
		else if (circle_type == right)
		{
			if (cross_right_up_flag && R_black[cross_right_up_line - 1] > Width / 2)
				circle_up_spot = cross_right_up_line;
		}
		if (circle_up_spot)
		{
			regression(circle_type, circle_up_spot, circle_up_spot - 3);
			draw_line(circle_type, High - 1, circle_up_spot, parameterB, parameterA);
			for (i = High - 1; i >= circle_up_spot; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
		}
		else
		{
			circle_protect--;
			if (circle_protect <= -20)
			{
				circle_state = 0;
				circle_type = 0;
				small_circle_flag = 0;
				circle_protect = 2;
			}
		}
	}
}
void cross_out(uint8 type)
{
	int i;
	if (type == left+2)
	{
		if (cross_left_up_flag&& cross_left_up_line>=30)
		{
			if (cross_left_up_flag && cross_right_down_flag)two_point_regression(cross_left_up_line, L_black[cross_left_up_line], cross_right_down_line, R_black[cross_right_down_line]);
			else if (cross_left_up_flag)two_point_regression(cross_left_up_line, L_black[cross_left_up_line], High - 1, R_black[High - 1]-40);
			draw_two_point(right, High - 1, cross_left_up_line, parameterk, parameterb);
			if (cross_left_down_flag)
			{
				two_point_regression(cross_left_down_line, L_black[cross_left_down_line], 0, 0);
				draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
			}
			else
			{
				two_point_regression(High - 1, L_black[High - 1], 0, 0);
				draw_two_point(left, High-1, 0, parameterk, parameterb);
			}
			for (i = cross_left_up_line; i >= 0; i--) { L_black[i] =0; R_black[i] = 0; }
		}
		if (cross_right_down_flag)
		{
			two_point_regression(cross_right_down_line,R_black[cross_right_down_line], 0, 0);
			draw_two_point(right, cross_right_down_line, 0, parameterk, parameterb);
			if (cross_left_down_flag)
			{
				two_point_regression(cross_left_down_line, L_black[cross_left_down_line], 0, 0);
				draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
			}
			else
			{
				two_point_regression(High - 1, R_black[High - 1], 0, 0);
				draw_two_point(left, High - 1, 0, parameterk, parameterb);
			}
		}
		else if (cross_right_up_flag)
		{
			two_point_regression(cross_right_up_line, L_black[cross_right_up_line], High - 1,Width-1);
			draw_two_point(right, High - 1, cross_right_up_line, parameterk, parameterb);
		}
		if (cross_left_down_flag)
		{
			two_point_regression(cross_left_down_line, L_black[cross_left_down_line], 0, 0);
			draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
		}
		//else*/
		/*{
			two_point_regression(High-1, Width-1, 0, 0);
			draw_two_point(right, High-1, 0, parameterk, parameterb);
			for (i = 0; i < High; i++)L_black[i] = 0;
		}*/
		/*cross_type_process();
		check_line();*/
	}
	else if (type == right+2)
	{
		if (cross_right_up_flag&& cross_right_up_line>=30)
		{
			if (cross_right_up_flag && cross_left_down_flag)two_point_regression(cross_right_up_line, R_black[cross_right_up_line], cross_left_down_line, L_black[cross_left_down_line]);
			else if (cross_right_up_flag)two_point_regression(cross_right_up_line, R_black[cross_right_up_line], High - 1, L_black[High - 1] + 40);
			draw_two_point(left, High - 1, cross_right_up_line, parameterk, parameterb);
			if (cross_right_down_flag)
			{
				two_point_regression(cross_right_down_line, R_black[cross_right_down_line], 0, Width - 1);
				draw_two_point(right, cross_right_down_line, 0, parameterk, parameterb);
			}
			else
			{
				two_point_regression(High - 1, R_black[High - 1], 0, Width - 1);
				draw_two_point(right, High-1, 0, parameterk, parameterb);
			}
			for (i = cross_left_up_line; i >= 0; i--) { L_black[i] = Width-1; R_black[i] = Width-1; }
		}
		if (cross_left_down_flag)
		{
			two_point_regression(cross_left_down_line, L_black[cross_left_down_line], 0, Width-1);
			draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
			if (cross_right_down_flag)
			{
				two_point_regression(cross_right_down_line, R_black[cross_right_down_line], 0, Width - 1);
				draw_two_point(right, cross_right_down_line, 0, parameterk, parameterb);
			}
			else
			{
				two_point_regression(High - 1, L_black[High - 1], 0, Width - 1);
				draw_two_point(right, High - 1, 0, parameterk, parameterb);
			}
		}
		else if (cross_left_up_flag)
		{
			two_point_regression(cross_left_up_line, M_black[cross_left_up_line], High - 1, 0);
			draw_two_point(left, High-1, cross_left_up_line, parameterk, parameterb);
		}
		if (cross_right_down_flag)
		{
			two_point_regression(cross_right_down_line, R_black[cross_right_down_line], 0, Width - 1);
			draw_two_point(right, cross_right_down_line, 0, parameterk, parameterb);
		}
		/*{
			two_point_regression(High - 1, 0, 0, Width-1);
			draw_two_point(left, High - 1, 0, parameterk, parameterb);
			for (i = 0; i < High; i++)R_black[i] = Width-1;
		}*/
		/*cross_type_process();
		check_line();*/
	}
	else if (type == left||type==right)
	{
		cross_type_process();
		check_line();
	}
}
void cross_process()
{
	int i;
	if (turn_direction <= 2)
	{
		if (cross_state == 1)
		{
			for (i = 1; i <= 4; i++)
			{
				eight_area_find_cross_spot(i);
			}
			if (cross_left_down_flag && cross_left_up_flag)
			{
				if (cross_left_down_line < cross_left_up_line + 5)
					cross_left_down_flag = 0;
				if (cross_right_down_flag && my_abs(cross_left_up_line, cross_right_down_line) > 60 && cross_right_down_line > cross_left_down_line)cross_left_down_flag = 0;
			}
			if (cross_right_down_flag && cross_right_up_flag)
			{
				if (cross_right_down_line < cross_right_up_line + 5)cross_right_down_flag = 0;
				if (cross_left_down_flag && my_abs(cross_left_up_line, cross_right_down_line) > 60 && cross_right_down_line < cross_left_down_line)cross_right_down_flag = 0;
			}
			if (!cross_left_down_flag && !cross_right_down_flag)
			{
				cross_state = 2;
			}
		}
		if (cross_state == 2)
		{
			for (i = 3; i <= 4; i++)
			{
				eight_area_find_cross_spot(i);
			}
			if (!cross_left_up_flag && turn_direction == right + 2)
			{
				for (i = High - 60; i >= 10; i--)
				{
					if (L_black[i] <= 5 && L_black[i + 1] <= 5 && L_black[i + 2] <= 5 && my_abs(L_black[i], L_black[i - 1] > 5) && L_black[i - 1] < L_black[i - 2] && L_black[i - 2] < L_black[i - 3])
					{
						cross_left_up_flag = 1;
						cross_left_up_line = i - 2;
						break;
					}
				}
			}
			if (!cross_right_up_flag && turn_direction == left + 2)
			{
				for (i = High - 60; i >= 10; i--)
				{
					if (R_black[i] >= Width - 5 && R_black[i + 1] >= Width - 5 && R_black[i + 2] >= Width - 5 && my_abs(R_black[i], R_black[i - 1] > 5) && R_black[i - 1] > R_black[i - 2] && R_black[i - 2] > R_black[i - 3])
					{
						cross_right_up_flag = 1;
						cross_right_up_line = i - 2;
						break;
					}
				}
			}
			if (!cross_right_up_flag && !cross_left_up_flag && (last_left_up >= 60 || last_right_up >= 60 || last_left_down >= 60 && last_right_down >= 60))
			{
				cross_state = 0;
				cross_flag = 0;
			}
		}
		cross_type_process();
		check_line();
	}
	else if (turn_direction == left + 2)
	{
		if (cross_state == 1)
		{
			check_flag = 0;
			static float temp_k = 0, temp_b = 0;
			cross_left_down_flag = 0; cross_left_up_flag = 0;
			cross_right_down_flag = 0; cross_right_up_flag = 0;
			cross_left_down_line = 0; cross_left_up_line = 0;
			cross_right_down_line = 0; cross_right_up_line = 0;
			uint8 state_change = 0;
			for (uint8 i = High - 10; i >= 30; i--)
			{
				if (!cross_left_down_flag)
				{
					if (L_black[i] >= L_black[i - 3] + 5 && my_abs(L_black[i + 1], L_black[i + 2] < 3) && my_abs(L_black[i + 2], L_black[i + 3] <= 3) && my_abs(L_black[i + 3], L_black[i + 4]) <= 3
						&& L_black[i - 3] < 5 && L_black[i - 4]<5 && road_width[i - 3]>Width / 2)
					{
						cross_left_down_flag = 1;
						cross_left_down_line = i;
					}
				}
				if (!cross_right_down_flag)
				{
					if (R_black[i] + 5 <= R_black[i - 3] && my_abs(R_black[i + 1], R_black[i + 2] < 3) && my_abs(R_black[i + 2], R_black[i + 3] <= 3) && my_abs(R_black[i + 3], R_black[i + 4]) <= 3
						&& R_black[i - 3] > Width - 5 && R_black[i - 4] > Width - 5 && road_width[i - 3] > Width / 2)
					{
						cross_right_down_flag = 1;
						cross_right_down_line = i;
					}
				}
				if (!cross_left_up_flag)
				{
					if (L_black[i + 2] < 5 && L_black[i + 3] < 5 && L_black[i + 4] < 5 && L_black[i] >= L_black[i + 3] + 5 && my_abs(L_black[i - 4], L_black[i - 5] )<=3 && my_abs(L_black[i - 5], L_black[i - 6])<=3 && my_abs(L_black[i - 6], L_black[i - 7]) <= 3)
					{
						cross_left_up_flag = 1;
						cross_left_up_line = i;
					}
				}
				if (!state_change)
				{
					if (R_black[i + 1] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5 && R_black[i - 1] + 5 <= R_black[i + 3] && R_black[i - 8] > Width - 5 && R_black[i - 9] > Width - 5 && my_abs(R_black[i - 1], R_black[i - 2]) < 3)
					{
						state_change = i;
					}
				}
			}
			if (cross_left_up_flag && cross_right_down_flag && cross_right_down_line <= 100)
			{
				two_point_regression(cross_left_up_line - 3, L_black[cross_left_up_line - 3], cross_right_down_line, R_black[cross_right_down_line]);
				draw_two_point(right, cross_left_down_line, 0, parameterk, parameterb);
			}
			else if (cross_left_up_flag)
			{
				two_point_regression(High - 1, Width - 1, cross_left_up_line - 5, L_black[cross_left_up_line - 5]);
				draw_two_point(right, High - 1, cross_left_up_line, parameterk, parameterb);
			}
			else if (cross_right_down_flag)
			{
				two_point_regression(0, 0, cross_right_down_line, R_black[cross_right_down_line]);
				draw_two_point(right, cross_left_down_line, 0, parameterk, parameterb);
			}
			if (cross_left_up_flag && cross_left_down_flag)
			{
				two_point_regression(cross_left_down_line, L_black[cross_left_down_line], cross_left_up_line - 5, 0);
				draw_two_point(left, High - 1, cross_left_up_line, parameterk, parameterb);
				temp_k = parameterk; temp_b = parameterb;
			}
			else if (cross_left_up_flag)
			{
				draw_two_point(left, High - 1, cross_left_up_line, temp_k, temp_b);
			}
			for (uint8 i = 0; i < High; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
			if (cross_left_up_flag)
			{
				for (uint8 i = 0; i < cross_left_up_line; i++)M_black[i] = 0;
			}
			for (i = High - 10; i > 60; i--)
			{
				if (R_black[i + 1] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5 && R_black[i - 1] + 5 < R_black[i + 1] && my_abs(R_black[i - 1], R_black[i - 2]) <= 3 && my_abs(R_black[i - 2], R_black[i - 3]) <= 3)
				{
					state_change = i;
					break;
				}
			}
			if (state_change)
			{
				cross_state = 2;
			}
		}
		else if (cross_state == 2)
		{
			uint8 i, j;
			uint8 flag = 0;
			for (i = 118; i > 0; i--)
			{
				L_black[i] = 0;
				for (j = 1; j < Width - 2; j++)
				{
					if (gray[i][j] != 0 && gray[i][j + 1] == 0 && gray[i][j + 2] == 0)
					{
						R_black[i] = j;
						break;
					}
				}
				if (j >= Width - 2)
				{
					R_black[i] = Width - 1;
				}
				M_black[i] = (L_black[i] + R_black[i]) / 2;
				if (M_black[i] < 5)break;
			}
			for (i = i; i > 0; i--)M_black[i] = 0;
			for (i = High - 10; i > 30; i--)
			{
				if (R_black[i + 1] > Width - 5 && R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5 && R_black[i - 1] + 5 < R_black[i + 1] && my_abs(R_black[i - 4], R_black[i - 5]) <= 3 && my_abs(R_black[i - 5], R_black[i - 6]) <= 3 && my_abs(R_black[i - 6], R_black[i - 7]) <= 3)
				{
					flag = i;
					break;
				}
			}
			if (flag)
			{
				two_point_regression(High - 1, R_black[High - 1], flag - 3, R_black[flag - 3]);
				draw_two_point(right, High - 1, flag, parameterk, parameterb);
				for (i = High - 1; i > flag; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			else
			{
				cross_state = 0;
				cross_flag = 0;
				turn_direction = 0;
				check_flag = 1;
			}
		}
	}
	else if (turn_direction == right + 2)
	{
		if (cross_state == 1)
		{
			check_flag = 0;
			static float temp_k = 0, temp_b = 0;
			cross_left_down_flag = 0; cross_left_up_flag = 0;
			cross_right_down_flag = 0; cross_right_up_flag = 0;
			cross_left_down_line = 0; cross_left_up_line = 0;
			cross_right_down_line = 0; cross_right_up_line = 0;
			uint8 state_change = 0;
			for (uint8 i = High - 10; i >= 30; i--)
			{
				if (!cross_left_down_flag)
				{
					if (L_black[i] >= L_black[i - 3] + 5 && my_abs(L_black[i + 1], L_black[i + 2] < 3) && my_abs(L_black[i + 2], L_black[i + 3] < 3) && my_abs(L_black[i + 3], L_black[i + 4]) < 3
						&& L_black[i - 3] < 5 && L_black[i - 4]<5 && road_width[i - 3]>Width / 2)
					{
						cross_left_down_flag = 1;
						cross_left_down_line = i;
					}
				}
				if (!cross_right_down_flag)
				{
					if (R_black[i] + 5 <= R_black[i - 3] && my_abs(R_black[i + 1], R_black[i + 2] < 3) && my_abs(R_black[i + 2], R_black[i + 3] < 3) && my_abs(R_black[i + 3], R_black[i + 4]) < 3
						&& R_black[i - 3] > Width - 5 && R_black[i - 4] > Width - 5 && road_width[i - 3] > Width / 2)
					{
						cross_right_down_flag = 1;
						cross_right_down_line = i;
					}
				}
				if (!cross_right_up_flag)
				{
					if (R_black[i + 2] > Width - 5 && R_black[i + 3] > Width - 5 && R_black[i + 4] > Width - 5 && R_black[i - 1] + 5 <= R_black[i + 1] && my_abs(R_black[i - 4], R_black[i - 5]) <= 3 && my_abs(R_black[i - 5], R_black[i - 6]) <= 3 && my_abs(R_black[i - 6], R_black[i - 7]) <= 3)
					{
						cross_right_up_flag = 1;
						cross_right_up_line = i;
					}
				}
				if (!state_change)
				{
					if (L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i + 3] < 5 && L_black[i - 1] >= L_black[i + 3] + 5 && L_black[i - 6] < 5 && L_black[i - 7] < 5 && my_abs(L_black[i - 1], L_black[i - 2]) < 3)
					{
						state_change = i;
					}
				}
			}
			if (cross_right_up_flag && cross_left_down_flag && cross_left_down_line <= 100)
			{
				two_point_regression(cross_right_up_line - 3, R_black[cross_right_up_line - 3], cross_left_down_line, L_black[cross_left_down_line]);
				draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
			}
			else if (cross_right_up_flag)
			{
				two_point_regression(High - 1, 0, cross_right_up_line - 5, R_black[cross_right_up_line - 5]);
				draw_two_point(left, High - 1, cross_right_up_line, parameterk, parameterb);
			}
			else if (cross_left_down_flag)
			{
				two_point_regression(0, Width - 1, cross_left_down_line+2, L_black[cross_left_down_line+2]);
				draw_two_point(left, cross_left_down_line, 0, parameterk, parameterb);
			}
			if (cross_right_up_flag && cross_right_down_flag)
			{
				two_point_regression(cross_right_down_line, R_black[cross_right_down_line], cross_right_up_line - 5, Width - 1);
				draw_two_point(right, cross_right_down_line, cross_right_up_line, parameterk, parameterb);
				temp_k = parameterk; temp_b = parameterb;
			}
			else if (cross_right_up_flag)
			{
				draw_two_point(right, High - 1, cross_right_up_line, temp_k, temp_b);
			}
			for (uint8 i = 0; i < High; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
			if (cross_right_up_flag)
			{
				for (uint8 i = 0; i < cross_right_up_line; i++)M_black[i] = Width - 1;
			}
			for (i = High - 10; i > 60; i--)
			{
				if (L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i + 3] < 5 && L_black[i - 1]  > L_black[i + 1]+5 && my_abs(L_black[i - 1], L_black[i - 2]) <= 3 && my_abs(L_black[i - 2], L_black[i - 3]) <= 3)
				{
					state_change = i;
					break;
				}
			}
			if (state_change)
			{
				cross_state = 2;
			}
		}
		else if (cross_state == 2)
		{
			uint8 i, j;
			uint8 flag = 0;
			for (i = 119; i > 0; i--)
			{
				R_black[i] = Width - 1;
				for (j = Width - 2; j > 2; j--)
				{
					if (gray[i][j] != 0 && gray[i][j - 1] == 0 && gray[i][j - 2] == 0)
					{
						L_black[i] = j;
						break;
					}
				}
				if (j <= 2)
				{
					L_black[i] = 0;
				}
				M_black[i] = (L_black[i] + R_black[i]) / 2;
				if (M_black[i] < 5)break;
			}
			for (i = i; i > 0; i--)M_black[i] = Width - 1;
			for (i = High - 10; i > 30; i--)
			{
				if (L_black[i + 1] < 5 && L_black[i + 2] < 5 && L_black[i + 3] < 5 && L_black[i - 1]  > L_black[i + 1] + 5 && my_abs(L_black[i - 1], L_black[i - 2]) < 3 && my_abs(L_black[i - 2], L_black[i - 3]) < 3)
				{
					flag = i;
					break;
				}
			}
			if (flag)
			{
				two_point_regression(High - 1, L_black[High - 1], flag - 3, L_black[flag - 3]);
				draw_two_point(left, High - 1, flag, parameterk, parameterb);
				for (i = High - 1; i > flag; i--)M_black[i] = (L_black[i] + R_black[i]) / 2;
			}
			else
			{
				cross_state = 0;
				cross_flag = 0;
				turn_direction = 0;
				check_flag = 1;
			}
		}
		printf("hgdfjzhsd:%d	%d	%d   state:%d\n", cross_left_down_line, cross_right_down_line, cross_right_up_line,cross_state);
	}
	/*if (cross_flag)
	{
		if (temp_position != cross_count)
		{
			cross_type_process();
			check_line();
		}
		else
		{
			cross_out(turn_direction);
			for (i = 0; i < High - 1; i++)M_black[i] = (L_black[i] + R_black[i]) / 2;
		}
	}*/
}
bool ramp_judge()
{
	bool judge = 0;
	uint8 i;
	for (i = 0; i < 5; i++)
	{
		if (road_width[i] < 5)break;
	}
	if (i >= 5)judge = 1;
	/*for (i = 30; i <= High - 40; i++)
	{
		if (L_black[i] > 5 && L_black[i - 3] > 5 && L_black[i - 5] > 5 && L_black[i + 3] > 5 && L_black[i + 5] > 5)
		{
			for (tempi = i - 5; tempi < i + 5; tempi++)
			{
				if (my_abs(L_black[tempi], L_black[tempi + 1]) >= 3)break;
			}
			if (tempi >= i + 5)
			{
				regression(left, i+5,i);
				k1 = parameterB;
				regression(left,i,i - 5);
				k2 = parameterB;
				if (my_abs(100.f, k1*1.0f/k2 * 100.f) > 100)left_line = i;
				if (left_line || right_line)break;
			}
		}
	}*/
	if (judge)printf("ramp_judge_ready");
	return judge;
}
int collinear(double a[2], double b[2], double c[2])//判断三点是否共线，共线返回1
{
	double k1, k2;
	double kx1, ky1, kx2, ky2;
	if (a[0] == b[0] && b[0] == c[0])  return 1;//三点横坐标都相等，共线
	else
	{
		kx1 = b[0] - a[0];
		kx2 = b[0] - c[0];
		ky1 = b[1] - a[1];
		ky2 = b[1] - a[1];
		k1 = ky1 / kx1;
		k2 = ky2 / kx2;
		if (k1 == k2) return 1;//AB与BC斜率相等，共线
		else  return 0;//不共线
	}
}
double curvature(double a[2], double b[2], double c[2])//double为数据类型，
{                                                    //数组a[2]为点a的坐标信息，a[0]为a的x坐标，a[1]为a的y坐标
	double cur;//求得的曲率
	if (collinear(a, b, c) == 1)//判断三点是否共线
	{
		cur = 0.0;//三点共线时将曲率设为某个值，0
	}
	else
	{
		double radius=0;//曲率半径
		double dis=0, dis1=0, dis2=0, dis3=0;//距离
		double cosA=0,sinA=0;//ab确定的边所对应的角A的cos值
		dis1 = sqrt((a[0]-b[0])*(a[0]-b[0])+ (a[1] - b[1]) * (a[1] - b[1]));
		dis2 = sqrt((a[0] - c[0]) * (a[0] - c[0]) + (a[1] - c[1]) * (a[1] - c[1]));
		dis3 = sqrt((b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1]));
		dis = dis2 * dis2 + dis3 * dis3 - dis1 * dis1;
		cosA = dis / (2 * dis2 * dis3);//余弦定理
		sinA = sqrt(1 - cosA * cosA);//求正弦
		radius = 0.5 * dis1 / sinA;
		cur = 1 / radius;
	}
	return cur;
}
void bin_filter()
{
	for (uint8 i = 1; i < High - 1; i++)
	{
		for (uint8 j = 1; j < Width - 1; j++)
		{
			if (!gray[i][j])
			{
				if (gray[i - 1][j] && gray[i + 1][j] && gray[i][j - 1]) {
					gray[i][j] = 1;
					continue;
				}
				if (gray[i - 1][j] && gray[i + 1][j] && gray[i][j + 1]) {
					gray[i][j] = 1;
					continue;
				}
				if (gray[i - 1][j] && gray[i][j - 1] && gray[i][j + 1]) {
					gray[i][j] = 1;
					continue;
				}
				if (gray[i + 1][j] && gray[i][j - 1] && gray[i][j + 1]) {
					gray[i][j] = 1;
					continue;
				}
			}
		}
	}
	for (uint8 i = 1; i < High - 1; i++)
	{
		for (uint8 j = 1; j < Width - 1; j++)
		{
			if (gray[i][j])
			{
				if (!gray[i - 1][j] && !gray[i + 1][j] && !gray[i][j - 1]) {
					gray[i][j] = 0;
					continue;
				}
				if (!gray[i - 1][j] && !gray[i + 1][j] && !gray[i][j + 1]) {
					gray[i][j] = 0;
					continue;
				}
				if (!gray[i - 1][j] && !gray[i][j - 1] && !gray[i][j + 1]) {
					gray[i][j] = 0;
					continue;
				}
				if (!gray[i + 1][j] && !gray[i][j - 1] && !gray[i][j + 1]) {
					gray[i][j] = 0;
					continue;
				}
			}
		}
	}
}
void ss_curve_judge()
{
	uint8 low = 0, up = 0;
	uint8 low2 = 0, up2 = 0;
	for (uint8 i = High - 10; i > 10; i--)
	{
		if (R_black[i] <= R_black[i + 3] && R_black[i] <= R_black[i - 3] && R_black[i] < R_black[i + 5] && R_black[i] < R_black[i - 5] && R_black[i]+3 < R_black[i - 7] && R_black[i]+3 < R_black[i + 7]
			&& my_abs(R_black[i], R_black[i + 3]) < 7 && my_abs(R_black[i + 3], R_black[i + 5]) < 7 && my_abs(R_black[i + 5], R_black[i + 7]) < 7
			&& my_abs(R_black[i], R_black[i - 3]) < 7 && my_abs(R_black[i - 3], R_black[i - 5]) < 7 && my_abs(R_black[i - 5], R_black[i - 7]) < 7)
		{
			low = i;
			break;
		}
		else if (L_black[i] >= L_black[i + 3] && L_black[i] >= L_black[i - 3] && L_black[i] > L_black[i + 5] && L_black[i] > L_black[i - 5] && L_black[i] > L_black[i - 7] + 3 && L_black[i] > L_black[i + 7] + 3
			&& my_abs(L_black[i], L_black[i + 3]) < 7 && my_abs(L_black[i + 3], L_black[i + 5]) < 7 && my_abs(L_black[i + 5], L_black[i + 7]) < 7
			&& my_abs(L_black[i], L_black[i - 3]) < 7 && my_abs(L_black[i - 3], L_black[i - 5]) < 7 && my_abs(L_black[i - 5], L_black[i - 7]) < 7)
		{
			low2 = i;
			break;
		}
	}
	for (uint8 i = High - 10; i > 10; i--)
	{
		if (R_black[i] >= R_black[i + 3] && R_black[i] >= R_black[i - 3] && R_black[i] > R_black[i + 5] && R_black[i] > R_black[i - 5] && R_black[i] > R_black[i - 7]+3 && R_black[i] > R_black[i + 7]+3
			&& my_abs(R_black[i], R_black[i + 3]) < 7 && my_abs(R_black[i + 3], R_black[i + 5]) < 7 && my_abs(R_black[i + 5], R_black[i + 7]) < 7
			&& my_abs(R_black[i], R_black[i - 3]) < 7 && my_abs(R_black[i - 3], R_black[i - 5]) < 7 && my_abs(R_black[i - 5], R_black[i - 7]) < 7)
		{
			up=i;
			break;
		}
		else if (L_black[i] <= L_black[i + 3] && L_black[i] <= L_black[i - 3] && L_black[i] < L_black[i + 5] && L_black[i] < L_black[i - 5] && L_black[i] + 3 < L_black[i - 7] && L_black[i] + 3 < L_black[i + 7]
			&& my_abs(L_black[i], L_black[i + 3]) < 7 && my_abs(L_black[i + 3], L_black[i + 5]) < 7 && my_abs(L_black[i + 5], L_black[i + 7]) < 7
			&& my_abs(L_black[i], L_black[i - 3]) < 7 && my_abs(L_black[i - 3], L_black[i - 5]) < 7 && my_abs(L_black[i - 5], L_black[i - 7]) < 7)
		{
			up2 = i;
			break;
		}
	}
	if (low && up && low > up + 8|| low2 && up2 && low2 > up2 + 8)
	{
		s_curve_flag = 1;
	}
	printf("low   %d   up:    %d\n", low, up);
}
void image_process()
{
	static int circle_protect = 2;
	uint8 cross_spot_count = 0;
	int i=0, j=0;
	uint8 area_select = 0;
	get_RGB();
	average_filter();
	dst_find_white();
	
	find_area();
	//
	uint8 temp_count = 0;
	uint8 line_count = 0;
	for (i = High - 1; i >= High - 10; i--)
	{
		for (j = 0; j < Width; j++)
		{
			if (cpy_gray[i][j])temp_count++;
		}
		if (temp_count >= 15)line_count++;
		temp_count = 0;
		if (line_count >= 5)
		{
			area_select = 1;
			break;
		}
	}//记录图像最底下是否被连通域抹除
	//
	if (area_select)
	{
		for (i = 0; i < High; i++)
		{
			for (j = 0; j < Width; j++)
			{
				gray[i][j] = cpy_gray[i][j];
			}
		}
	}
	bin_filter();
	if (circle_type)circle_find_line();
	else if (cross_flag)find_max_list();
	else find_line();
	/*if (circle_type)circle_find_line();
	else if (cross_flag)find_max_list();
	else find_line();*/
//	L_R_nodown_judge(right);
	//L_R_circle_judge();
//调试区	
 	//   zera_judge();
	//ramp_judge();
	//L_R_circle_judge();
	/*circle_type = 1;
	//L_R_nodown_judge(right);
	int ls = my_line_offset(1, High - 20, 40);
	int rs = my_line_offset(2, High - 20, 40);
	circle_mid_find(circle_type);
	circle_down_find(circle_type);
	circle_up_find(circle_type);
	circle_find_leave_spot(circle_type);
	printf("judge:   %d  %d\n", ls, rs);*/
	/*find_cross_spot();
	cross_type_process();
	check_line();*/
	/*barrier_judge();
	if (barrier)
	{
		barrier_search();
	}*/
//
// 	   
	zera_flag = zera_judge();
	if (!cross_flag)
	{
		if (!circle_type)
		{
			if (zera_flag)check_line();
			if (!zera_flag && !barrier)barrier_judge();
		}
		if (!circle_type && !zera_flag && !barrier)L_R_circle_judge();
		if (circle_type)circle_process();
		printf("state:   %d\n", circle_state);
		if (barrier)
		{
			barrier_search();
		}
	}
	if (!cross_flag && !circle_type && !zera_flag && !barrier )
	{
		find_max_list();
		for (i = 1; i <= 4; i++)
		{
			eight_area_find_cross_spot(i);//1,2,3,4,左下1，右下，左上3，右上4
		}
		if (cross_left_up_flag && cross_right_up_flag || cross_left_down_flag && cross_right_down_flag)
		{
			cross_flag = 1;
			cross_state = 1;
			turn_direction = position_cross[cross_count++];
		}
		else 
		{
			turn_direction = 0;
			///*find_cross_spot();*/
			cross_type_process();
			//check_line();
		}
	}
	if (cross_flag)cross_process();
	if (barrier)printf("barrier  up:%d   down:%d   type:%d\n", barrier_up, barrier_down,barrier_type);
	if (circle_state)printf("type:%d  state:%d up:%d mid:%d down:%d leave:%d\n",circle_type,circle_state,circle_up_spot,circle_up_spot,circle_down_spot,circle_leave_spot);
	if (small_circle_flag)printf("small_circle\n");
	check_line();
	ss_curve_judge();
	curve_flag = curve_judge();
	if (s_curve_flag && !curve_flag)s_curve_flag = 0;
	//if(s_curve_flag)printf("end_line: sssssscurve   %d\n", scan_endline);
	if(curve_flag)printf("curve!!!!!!!!!!!!!!!!\n");
	printf("cross_count:%d\n", cross_count);

	/*for (i = 0; i < High; i++)
	{
		for (j = 0; j < Width; j++)
		{
			//image_copy[i][j] = ImageUsed[i][j];//逆透视变换效果图
			if (cpy_gray[i][j])image_copy[i][j] = 0xffffff;
			else image_copy[i][j] = 0;
		}
	}*/
}
/********************************************************************************************************************/
void draw_test()
{
	int i, j;
	for (i = 0; i < 120; i++)
	{
		for (j = 0; j < 160; j++)
		{
			if (gray_mode==0)
			{
				test.at<Vec3b>(i, j)[0] = image_copy[i][j];
				test.at<Vec3b>(i, j)[1] = image_copy[i][j] >> 8;
				test.at<Vec3b>(i, j)[2] = image_copy[i][j] >> 16;
			}
			else
			{
				test.at<Vec3b>(i, j)[0] = image_copy[i][j];
				test.at<Vec3b>(i, j)[1] = image_copy[i][j] >> 8;
				test.at<Vec3b>(i, j)[2] = image_copy[i][j] >> 16;
			}
		}
	}
}
void scc8660_image_get(Mat frame)
{
	int i, j,k;
	for (i = 0; i < 120; i++)
	{
		for (j = 0; j < 160; j++) {
				scc8660_image[i][j] = (int)(frame.at<Vec3b>(i, j)[2]<<16)+ (int)(frame.at<Vec3b>(i, j)[1] <<8)+ (int)(frame.at<Vec3b>(i, j)[0]);
		}
	}
}
void my_draw_line()
{
	int i, j;
	for (i = 0; i < 120; i++)
	{
		/*for (j = 0; j < 120; j++) {
			test.at<Vec3b>(j, i)[0] = 0;
			test.at<Vec3b>(j, i)[1] = 0;
			test.at<Vec3b>(j, i)[2] = 255;
		}*/
		test.at<Vec3b>(i,L_black[i])[0] = 0;
		test.at<Vec3b>(i,L_black[i])[1] = 255;
		test.at<Vec3b>(i,L_black[i])[2] = 255;
		test.at<Vec3b>(i,R_black[i])[0] = 255;
		test.at<Vec3b>(i,R_black[i])[1] = 0;
		test.at<Vec3b>(i,R_black[i])[2] = 255;
		test.at<Vec3b>(i,M_black[i])[0] = 255;
		test.at<Vec3b>(i,M_black[i])[1] = 255;
		test.at<Vec3b>(i,M_black[i])[2] = 0;
	}
	
	//横向扫线显示
	/*for (i = 0; i < Width; i++)
	{
		test.at<Vec3b>(verti_L_black[i],i)[0] = 0;
		test.at<Vec3b>(verti_L_black[i],i)[1] = 255;
		test.at<Vec3b>(verti_L_black[i],i)[2] = 255;
		test.at<Vec3b>(verti_R_black[i],i)[0] = 255;
		test.at<Vec3b>(verti_R_black[i],i)[1] = 0;
		test.at<Vec3b>(verti_R_black[i],i)[2] = 255;
		test.at<Vec3b>(verti_M_black[i],i)[0] = 255;
		test.at<Vec3b>(verti_M_black[i],i)[1] = 255;
		test.at<Vec3b>(verti_M_black[i],i)[2] = 0;
	}*/
	//纵向扫线显示
}
void my_process::creat_button() {
	putText(output, "<", Point(800 / 2.5+50-100, 100-25), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 4, LINE_AA);
	putText(output, "+", Point(800 / 2.5 + 50, 100-25), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 4, LINE_AA);
	putText(output, ">", Point(800 / 2.5 + 50+100, 100-25), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 4, LINE_AA);
	Rect button(800-250, 25, 250, 100);
	rectangle(output, button, Scalar(255, 255, 255)-Scalar(100,100,100), -1, LINE_AA, 0);
	if(mode==0)putText(output, "location", Point(800-250, 75), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,0,0), 4, LINE_AA);
	else if(mode==1)putText(output, "photest", Point(800 - 250, 75), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 4, LINE_AA);
}
Point get_point(-1, -1);
Point button(-1, -1);
void static point_get(int event,int x,int y,int flags,void *userdata)
{
	if (event == EVENT_MOUSEMOVE) {
		get_point.x = x;
		get_point.y = y;
	}
	else if (event == EVENT_FLAG_LBUTTON)
	{
		button.x = x;
		button.y = y;
	}
}
void int_turn_char(char *a,int b)
{
	int i=0,length=0;
	char temp;
	do {
		a[i] = b % 10+'0';
		i++;
	} while (b /= 10);
	a[i] = '\0';
	length = i;
	for (i = 0; i < length / 2; i++)
	{
		temp = a[i];
		a[i] = a[length - 1 - i];
		a[length - 1 - i] = temp;
	}
}
char px[8], py[8];
void my_process::blackground_init( ) {
	int key;
	char now_po[11];
	Mat ph_RGB= Mat::zeros(Size(1600, 700), CV_8UC3);
	Vec3b p1,p2;	
	Mat temp,cpy;
	int j;
	int num = 0;
	Mat bigph;
	int k = 0;
	for (j = 0; j < 120; j++)
	{
		L_black[j] = 0;
		R_black[j] = 159;
		M_black[j] = 80;
	}
	Size dsize = Size(800, 600);
	temp = Mat::zeros(Size(800, 100), CV_8UC3);
	if(mode==0)output = Mat::zeros(Size(800,100), CV_8UC3);
	else if (mode==1)output = Mat::zeros(Size(1600, 100), CV_8UC3);
	test = Mat::zeros(Size(160, 120), CV_8UC3);
	if(mode==1)my_draw_line();
	namedWindow("上位机");
	resizeWindow("上位机", 1600, 700);
	creat_button();
	
	VideoCapture capture(address);
	ImagePerspective_Init();
	int afps=0;
	capture.open(address);
	if (!capture.isOpened()||capture.get(CAP_PROP_FRAME_WIDTH)!=160|| capture.get(CAP_PROP_FRAME_HEIGHT) != 120) {
		cout << "视频打开失败" << endl;
		return;
	}
	int scount;
	if (sum_fp < 0)scount = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));
	else scount = sum_fp;
	if (i >= scount-1)i = scount-1;
	capture.set(CAP_PROP_POS_FRAMES, i);
	capture.read(frame);
	resize(frame, bigph, dsize, 0, 0, INTER_AREA);
	if (mode == 0)
	{
		vconcat(bigph, output, temp);
	}
	else if (mode == 1)
	{
		if(showph)draw_test();
		my_draw_line();
		resize(test, cpy, dsize, 0, 0, INTER_AREA);
		hconcat(bigph, cpy, temp); vconcat(temp, output, temp);
	}imshow("上位机",temp);
	int length = 0;
	int fps_cpy = 0;
	j = 0;
	while (true)
	{
		key = waitKey(10);
		if (key == 97||key==100)
		{
			
			if (key == 97)
			{
				i--;
				if (i < 1)i = 1;
			}
			else if (key == 100)
			{
				i++;
				if (i > scount-1)i = scount - 1;
			}
			capture.set(CAP_PROP_POS_FRAMES,i);
			capture.read(frame);
			imwrite("test.bmp", frame);
			if (i > 1)scc8660_image_get(frame);
			if (key==27|| frame.empty())
			{
				capture.release();
				destroyAllWindows();
				return;
			}
			if (mode == 1)
			{
				image_process();
				
			}
			resize(frame, bigph, dsize, 0, 0, INTER_AREA);
			if (mode == 0)
			{
				vconcat(bigph, output, temp);
			}
			else if (mode == 1)
			{	
				draw_test();
				my_draw_line();
				test.copyTo(cpy);
				resize(test, cpy, dsize, 0, 0, INTER_AREA);
				hconcat(bigph, cpy, temp); vconcat(temp, output, temp);
			}
			int_turn_char(fps,i);
			putText(temp, "fp", Point(0, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(temp, "|", Point(300-5, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			for (num = 0; num < i * 8 / (scount-1); num++)now_po[num] = '-'; now_po[num] = '\0';
			putText(temp,now_po, Point(300, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(temp, "|", Point(300+200, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(temp, fps, Point(50,625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 1, LINE_AA);
			imshow("上位机", temp);
		}
		setMouseCallback("上位机",point_get,NULL);
		if (get_point.x > 0 && get_point.y > 0&&i>0&&get_point.x<800&&get_point.y<600) {
			p1 = temp.at<Vec3b>(get_point.y, get_point.x);
			p2 = temp.at<Vec3b>(get_point.y, get_point.x+800);
			temp.copyTo(ph_RGB);
			putText(ph_RGB, "fp", Point(0, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, fps, Point(50, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(ph_B,p1[0]); int_turn_char(ph_G, p1[1]); int_turn_char(ph_R, p1[2]);
			putText(ph_RGB, "B", Point(0, 625+25*3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);
			putText(ph_RGB, "G", Point(0, 625 + 25*2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, LINE_AA);
			putText(ph_RGB, "R", Point(0, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
			putText(ph_RGB, ph_B, Point(25, 625+25*3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_G, Point(25, 625 + 25*2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_R, Point(25, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(ph_B, p2[0]); int_turn_char(ph_G, p2[1]); int_turn_char(ph_R, p2[2]);
			putText(ph_RGB, "B", Point(800, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);
			putText(ph_RGB, "G", Point(800, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, LINE_AA);
			putText(ph_RGB, "R", Point(800, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
			putText(ph_RGB, ph_B, Point(825, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_G, Point(825, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_R, Point(825, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(px, get_point.x/5); int_turn_char(py,get_point.y/5);
			putText(ph_RGB, "y", Point(100, 625+25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, "x", Point(100, 625 + 25*2 ), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, py, Point(100+25, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, px, Point(100+25, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			imshow("上位机", ph_RGB);
		}
		else if (get_point.x > 800 && get_point.y > 0 && i > 0 && get_point.x < 800+800 && get_point.y < 600) {
			p2 = temp.at<Vec3b>(get_point.y, get_point.x);
			p1 = temp.at<Vec3b>(get_point.y, get_point.x-800);
			temp.copyTo(ph_RGB);
			int_turn_char(ph_B, p1[0]); int_turn_char(ph_G, p1[1]); int_turn_char(ph_R, p1[2]);
			putText(ph_RGB, "B", Point(0, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);
			putText(ph_RGB, "G", Point(0, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, LINE_AA);
			putText(ph_RGB, "R", Point(0, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
			putText(ph_RGB, ph_B, Point(25, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_G, Point(25, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_R, Point(25, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, "fp", Point(800, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, fps, Point(800+50, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(ph_B, p2[0]); int_turn_char(ph_G, p2[1]); int_turn_char(ph_R, p2[2]);
			putText(ph_RGB, "B", Point(800, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, LINE_AA);
			putText(ph_RGB, "G", Point(800, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1, LINE_AA);
			putText(ph_RGB, "R", Point(800, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1, LINE_AA);
			putText(ph_RGB, ph_B, Point(825, 625 + 25 * 3), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_G, Point(825, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, ph_R, Point(825, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(px, (get_point.x-800) / 5); int_turn_char(py, get_point.y / 5);
			putText(ph_RGB, "y", Point(800+100, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, "x", Point(800+100, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, py, Point(800 + 25+100, 625 + 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(ph_RGB, px, Point(800 + 25+100, 625 + 25 * 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			imshow("上位机", ph_RGB);
		}
		else if (button.y > 634 && button.y < 687)
		{
			if (button.x > 463 && button.x < 525)
			{
				i += scount / 8;
			}
			else if (button.x > 370 && button.x < 421)
			{
				i=1;
			}
			else if (button.x > 271 && button.x < 326)
			{
				i -=scount/8;
			}
			if (i < 1)i =1; if (i > scount - 1)i = scount - 1;
			capture.set(CAP_PROP_POS_FRAMES, i);
			capture.read(frame);
			if (i > 1)scc8660_image_get(frame);
			if (key == 27 || frame.empty())
			{
				capture.release();
				destroyAllWindows();
				return;
			}
			if (mode == 1)image_process();
			resize(frame, bigph, dsize, 0, 0, INTER_AREA);
			if (mode == 0)
			{
				vconcat(bigph, output, temp);
			}
			else if (mode == 1)
			{
				draw_test();
				my_draw_line();
				test.copyTo(cpy);
				resize(test, cpy, dsize, 0, 0, INTER_AREA);
				hconcat(bigph, cpy, temp); vconcat(temp, output, temp);
			}
			putText(temp, "fp", Point(0, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(temp, "|", Point(300 - 5, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			for (num = 0; num < i * 8 / (scount - 1); num++)now_po[num] = '-'; now_po[num] = '\0';
			putText(temp, now_po, Point(300, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			putText(temp, "|", Point(300 + 200, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			int_turn_char(fps, i);
			putText(temp, fps, Point(50, 625), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 1, LINE_AA);
			imshow("上位机", temp);
			button.x = -1; button.y = -1;
		}
		if (key == 27) {
				capture.release();
				destroyAllWindows();
				break;
		}
	}
}
