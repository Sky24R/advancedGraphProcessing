//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/imgproc.hpp"
//#include <opencv2/imgproc/types_c.h>
//using namespace std;
//using namespace cv;
////打印直方图
//Mat showHist(Mat img) {
//    int channels = 0;
//    MatND dstHist;
//    //设定像素取值范围
//    int hisSize[] = { 256 };//将数值分组，每个灰度范围一组
//    float midRanges[] = { 0,255 };
//    const float* ranges[] = { midRanges };
//    //计算直方图
//    calcHist(&img, 1, &channels, Mat(), dstHist, 1, hisSize, ranges, true, false);
//    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
//    double MaxValue;
//    minMaxLoc(dstHist, 0, &MaxValue, 0, 0);//图像最小最大值
//
//    normalize(dstHist, dstHist, 0, drawImage.rows, NORM_MINMAX, -1, Mat());
//
//    for (int i = 0; i < 256; i++) {
//        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / MaxValue);//四舍五入
//        //在直方图画布上画出直方图
//        line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 255, 255));
//    }
//    return drawImage;
//}
//
//
//Mat equalizeHist(Mat& image) {
//    Mat grayImage = image.clone();
//    int gray_sum = 0;//像素总数
//    int gray[256] = { 0 };//记录每个灰度级别下的像素个数
//    double gray_rate[256] = { 0 };//记录灰度分布密度
//    gray_sum = grayImage.rows * grayImage.cols;
//    //统计每个灰度下的像素个数
//    for (int i = 0; i < grayImage.rows; i++) {
//        uchar* p = grayImage.ptr<uchar>(i);
//        for (int j = 0; j < grayImage.cols; j++) {
//            int value = p[j];
//            gray[value]++;
//        }
//    }
//    //统计灰度频率
//    for (int i = 0; i < 256; i++) {
//        gray_rate[i] = ((double)gray[i] / gray_sum);
//    }
//
//    double gray_distribution[256] = { 0 };//记录累计密度
//    int gray_equal[256] = { 0 };//均衡化后的灰度值
//    //计算累计密度
//    gray_distribution[0] = gray_rate[0];
//    for (int i = 1; i < 256; i++) {
//        gray_distribution[i] = gray_distribution[i - 1] + gray_rate[i];
//    }
//    //重新计算均衡化后的灰度值，四舍五入。公式=（L-1）*T（这里变换一下公式变换为(L-1)）
//    for (int i = 0; i < 256; i++) {
//        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
//    }
//
//    //直方图均衡化，更新原图每个像素点的值
//    for (int i = 0; i < grayImage.rows; i++) {
//        uchar* p = grayImage.ptr<uchar>(i);
//        for (int j = 0; j < grayImage.cols; j++) {
//            p[j] = gray_equal[p[j]];
//        }
//    }
//    
//    return grayImage;
//}
//
//Mat colorHist(Mat& image) {
//    Mat img = image.clone();
//    //存储彩色直方图和图像通道向量
//    Mat colorImage;
//    vector<Mat> BGR_plane;
//    //分离BGR通道
//    split(img, BGR_plane);
//    //分别对BGR通道进行直方图均衡化
//    for (int i = 0; i < BGR_plane.size(); i++) {
//        BGR_plane[i]=equalizeHist(BGR_plane[i]);
//    }
//    //合并通道
//    merge(BGR_plane, colorImage);
//    //imshow("col", colorImage);
//    //colorImage = showHist(colorImage);
//    return colorImage;
//   
//}
//
//int main(int argc, char* argv[]) {
//    // 图片路径
//    Mat img = imread("D://1.jpg");
//    if (img.empty()) {
//        printf("could not load image...\n");
//        return -1;
//    }
//    imshow("img", img);
//    Mat gray, result1,result2, result3,eqImage,eqcolor;
//    cvtColor(img, gray, COLOR_BGR2GRAY);//先转为灰度图
//    result1 = showHist(gray);//计算灰度直方图
//    imshow("gray", gray);
//    imshow("gray_hist", result1);
//
//    eqImage = equalizeHist(gray);//直方图均衡化
//    result2 = showHist(eqImage);
//    imshow("eqImage", eqImage);
//    imshow("result2", result2);
//
//    eqcolor = colorHist(img);//彩色直方图
//    result3 = showHist(eqcolor);
//    imshow("eqcolor", eqcolor);
//    imshow("result3", result3);
//    waitKey(0);
//    return 0;
//}
