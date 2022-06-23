#include <stdio.h>  
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include<iostream>

using namespace std;
using namespace cv;




// 傅里叶变换
Mat imageDFT(Mat& src_image) {
    Mat src, fourier;
    Mat image = src_image;
    // 实部：图像 , 虚部：全部用0填充
    Mat re_im[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32FC1) };
    // 将实部与虚部合并，形成一个复数
    merge(re_im, 2, src);
    // 离散傅里叶变换
    dft(src, fourier);
    return fourier;
}
Mat showDFT(Mat zero) {
    Mat src = zero.clone();
    Mat fourier = imageDFT(zero);

    Mat plane[] = { Mat_<float>(src), Mat::zeros(src.size() , CV_32FC1) }; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
    split(fourier, plane);
    Mat tempu;
    vector<Mat> res;//
    // 获取未移动时的频谱
    magnitude(plane[0], plane[1], tempu);
    tempu += Scalar::all(1);
    log(tempu, tempu);
    // 归一化操作
    normalize(tempu, tempu, 1, 0, CV_MINMAX);//傅里叶频谱tempu
    return tempu;
}

// 傅里叶逆变换
Mat imageIDFT(Mat& fourier) {
    Mat invfourier;
    idft(fourier, invfourier, 0);
    Mat re_im[2];
    // 分离傅里叶变换的实部与虚部
    split(invfourier, re_im);
    normalize(re_im[0], re_im[0], 0, 1, CV_MINMAX);
    namedWindow("invfourier", WINDOW_NORMAL);
    imshow("invfourier", re_im[0]);
    waitKey(1000);
    return re_im[0];
}

// 零填充
Mat zeroPadding(Mat src) {
    int M = src.rows, N = src.cols;
    Mat img = Mat::zeros(Size(2 * N, 2 * M), CV_8UC1);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            img.at<uchar>(i, j) = src.at<uchar>(i, j);
        }
    }
    return img;
}
// 移动到中心
void shiftCenter(Mat& mat) {
    int cx = mat.cols / 2;
    int cy = mat.rows / 2;
    //元素坐标表示为(cx,cy)
    Mat part1(mat, Rect(0, 0, cx, cy));
    Mat part2(mat, Rect(cx, 0, cx, cy));
    Mat part3(mat, Rect(0, cy, cx, cy));
    Mat part4(mat, Rect(cx, cy, cx, cy));
    Mat temp;
    // 位置交换
    part1.copyTo(temp); //左上与右下交换位置
    part4.copyTo(part1);
    temp.copyTo(part4);
    part2.copyTo(temp); //右上与左下交换位置
    part3.copyTo(part2);
    temp.copyTo(part3);
}
// 移动傅里叶变换,将低频从四个角,移动到中心
vector<Mat> moveFourier(Mat zero) {
    Mat src = zero.clone();
    Mat fourier = imageDFT(zero);
    namedWindow("src", WINDOW_NORMAL);
    imshow("src", src);
    waitKey(1000);
    Mat plane[] = { Mat_<float>(src), Mat::zeros(src.size() , CV_32FC1) }; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
    split(fourier, plane);
    Mat tempu;

    // 获取未移动时的频谱
    magnitude(plane[0], plane[1], tempu);
    tempu += Scalar::all(1);
    log(tempu, tempu);
    // 归一化操作
    normalize(tempu, tempu, 1, 0, CV_MINMAX);//傅里叶频谱tempu
    namedWindow("傅里叶频谱", WINDOW_NORMAL);

    imshow("傅里叶频谱", tempu);
    waitKey(1000);
    // 以下的操作是移动图像  (零频移到中心)
    shiftCenter(plane[0]);  // 实部
    shiftCenter(plane[1]);  // 虚部
    // 存入向量
    Mat temp0 = plane[0].clone();
    Mat temp1 = plane[1].clone();
    vector<Mat> temp;  //移动中心
    temp.push_back(temp0);
    temp.push_back(temp1);

    // 获取原始图像的频谱图
    magnitude(plane[0], plane[1], plane[0]);
    plane[0] += Scalar::all(1);
    log(plane[0], plane[0]);
    // 归一化操作便于显示
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);//中心化的傅里叶频谱plane[0]
    namedWindow("中心化的傅里叶频谱", WINDOW_NORMAL);

    imshow("中心化的傅里叶频谱", plane[0]);
    waitKey(1000);
    return temp;
}

//计算Mat类型的指数
void MatPow(Mat& src, double exp) {
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            src.at<float>(i, j) = pow(src.at<float>(i, j), exp);
}

//移动图像到中心位置
Mat moveImage(Mat gray,Mat ifourier) {

    Mat src_image = ifourier;
    int row = gray.rows, col =gray.cols;
    Mat dst_image = src_image(Rect(row / 2, col / 2, col, row));
    return dst_image;

}

//寻找矩阵最大值:
double findMatMax(Mat& src) {
    double max = 0;
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            if (src.at<float>(i, j) > max)
                max = src.at<float>(i, j);
        }
    return max;
}

// 数据类型转换
void Mat_convert2int(Mat& src,Mat& dst) {
    double value;
    double max = findMatMax(src);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            value = 255 * src.at<float>(i, j) / max;
            if (value > 255)
                value = 255;
            if (value < 0)
                value = 0;
            dst.at<uchar>(i, j) = int(value);
        }
}

//生成理想低通滤波器
Mat makeIdealLowPassFilterKernel(int row, int col, double d0 = 160) {
    Mat ideal_low_pass_filter;
    int centeri = row / 2, centerj = col / 2;
    ideal_low_pass_filter = Mat::zeros(row, col, CV_32F);
    for (int i = 0; i < ideal_low_pass_filter.rows; i++)
        for (int j = 0; j < ideal_low_pass_filter.cols; j++) {
            double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
            if (d < d0)
                ideal_low_pass_filter.at<float>(i, j) = 1;
        }
    return ideal_low_pass_filter;
}

//生成理想高通滤波器
Mat makeIdealHighPassFilterKernel(int row, int col, double d0 = 10) {
    Mat ideal_high_pass_filter;
    int centeri = row / 2, centerj = col / 2;
    ideal_high_pass_filter = Mat::zeros(row, col, CV_32F);
    for (int i = 0; i < ideal_high_pass_filter.rows; i++)
        for (int j = 0; j < ideal_high_pass_filter.cols; j++) {
            double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
            if (d > d0)
                ideal_high_pass_filter.at<float>(i, j) = 1;
        }
    return ideal_high_pass_filter;
}
// 布特沃斯低通
Mat makeButterworseLowPassFilterKernel(int row, int col, int n = 2, double d0 = 60) {
    Mat butter_low_pass_filter;
    int centeri = row / 2, centerj = col / 2;
    butter_low_pass_filter = Mat::zeros(row, col, CV_32F);
    for (int i = 0; i < butter_low_pass_filter.rows; i++)
        for (int j = 0; j < butter_low_pass_filter.cols; j++) {
            double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
            butter_low_pass_filter.at<float>(i, j) = 1 / (1 + pow(d / d0, n));
        }
    return butter_low_pass_filter;
}
// 布特沃斯高通
Mat makeButterworseHighPassFilterKernel(int row, int col, int n = 2, double d0 = 10) {
    Mat butter_high_pass_filter;
    int centeri = row / 2, centerj = col / 2;
    butter_high_pass_filter = Mat::zeros(row, col, CV_32F);
    for (int i = 0; i < butter_high_pass_filter.rows; i++)
        for (int j = 0; j < butter_high_pass_filter.cols; j++) {
            double d = sqrt(pow(i - centeri, 2) + pow(j - centerj, 2));
            butter_high_pass_filter.at<float>(i, j) = 1 / (1 + pow(d0 / d, n));
        }
    return butter_high_pass_filter;
}

//频域滤波
Mat frequencyDomainFilter(Mat src, int select = 0) {
    Mat filter_kernel;
    Mat re, im;
    Mat blur_r, blur_i, blur;
    Mat zero = zeroPadding(src);//灰度图片零填充
    int row =zero.rows, col = zero.cols;
    vector<Mat> res = moveFourier(zero);//
    re = res[0].clone();
    im = res[1].clone();
    //生成滤波器
    if (select == 0)
        filter_kernel = makeIdealLowPassFilterKernel(row, col,160);
    else if (select == 1)
        filter_kernel = makeIdealHighPassFilterKernel(row, col, 10);
    else if (select == 2)
        filter_kernel = makeButterworseLowPassFilterKernel(row, col, 2, 60);
    else if (select == 3)
        filter_kernel = makeButterworseHighPassFilterKernel(row, col, 2, 10);
    // 相乘,滤波
    multiply(re, filter_kernel, blur_r);
    multiply(im, filter_kernel, blur_i);
    Mat plane1[] = { blur_r, blur_i };
    // 实部与虚部合并
    merge(plane1, 2, blur);

    //转换回相应的象限
    shiftCenter(blur);

    Mat invfourier = imageIDFT(blur);
    int cx = invfourier.cols / 2;
    int cy = invfourier.rows / 2;
    Mat dst(invfourier, Rect(0, 0, cx, cy));
    /*namedWindow("dst", WINDOW_NORMAL);
    imshow("dst", dst);
    waitKey(1000);*/

    return dst;
}



int main() {
	Mat img = imread("D://1.jpg");
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
    //imshow("gray", gray);
    //waitKey(1000);

    //Mat fourier = showDFT(gray);//傅里叶变换
    //imshow("fourier", fourier);
    //waitKey(1000);

    //
    //Mat tep = imageDFT(gray);
    //Mat ifourier = imageIDFT(tep);//逆傅里叶变换
    //imshow("ifourier", ifourier);
    //waitKey(1000);


 
    //Mat ideal_low_pass_filte = frequencyDomainFilter(gray,0);//理想低通滤波
    //namedWindow("ideal_low_pass_filte", WINDOW_NORMAL);
    //imshow("ideal_low_pass_filte", ideal_low_pass_filte);
    //waitKey(1000);
    //Mat ideal_high_pass_filter = frequencyDomainFilter(gray, 1);//理想高通滤波
    //namedWindow("ideal_high_pass_filter", WINDOW_NORMAL);
    //imshow("ideal_high_pass_filter", ideal_high_pass_filter);
    //waitKey(1000);
    //Mat butter_low_pass_filter = frequencyDomainFilter(gray, 2);//布特沃斯低通
    //namedWindow("butter_low_pass_filter", WINDOW_NORMAL);
    //imshow("butter_low_pass_filter", butter_low_pass_filter);
    //waitKey(1000);
    Mat butter_high_pass_filter = frequencyDomainFilter(gray, 3);//布特沃斯高通
    namedWindow("butter_high_pass_filter", WINDOW_NORMAL);
    imshow("butter_high_pass_filter", butter_high_pass_filter);
    waitKey(1000);

}