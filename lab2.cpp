//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "opencv2/imgproc.hpp"
//#include <opencv2/imgproc/types_c.h>
//using namespace std;
//using namespace cv;
////��ӡֱ��ͼ
//Mat showHist(Mat img) {
//    int channels = 0;
//    MatND dstHist;
//    //�趨����ȡֵ��Χ
//    int hisSize[] = { 256 };//����ֵ���飬ÿ���Ҷȷ�Χһ��
//    float midRanges[] = { 0,255 };
//    const float* ranges[] = { midRanges };
//    //����ֱ��ͼ
//    calcHist(&img, 1, &channels, Mat(), dstHist, 1, hisSize, ranges, true, false);
//    Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
//    double MaxValue;
//    minMaxLoc(dstHist, 0, &MaxValue, 0, 0);//ͼ����С���ֵ
//
//    normalize(dstHist, dstHist, 0, drawImage.rows, NORM_MINMAX, -1, Mat());
//
//    for (int i = 0; i < 256; i++) {
//        int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / MaxValue);//��������
//        //��ֱ��ͼ�����ϻ���ֱ��ͼ
//        line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 255, 255));
//    }
//    return drawImage;
//}
//
//
//Mat equalizeHist(Mat& image) {
//    Mat grayImage = image.clone();
//    int gray_sum = 0;//��������
//    int gray[256] = { 0 };//��¼ÿ���Ҷȼ����µ����ظ���
//    double gray_rate[256] = { 0 };//��¼�Ҷȷֲ��ܶ�
//    gray_sum = grayImage.rows * grayImage.cols;
//    //ͳ��ÿ���Ҷ��µ����ظ���
//    for (int i = 0; i < grayImage.rows; i++) {
//        uchar* p = grayImage.ptr<uchar>(i);
//        for (int j = 0; j < grayImage.cols; j++) {
//            int value = p[j];
//            gray[value]++;
//        }
//    }
//    //ͳ�ƻҶ�Ƶ��
//    for (int i = 0; i < 256; i++) {
//        gray_rate[i] = ((double)gray[i] / gray_sum);
//    }
//
//    double gray_distribution[256] = { 0 };//��¼�ۼ��ܶ�
//    int gray_equal[256] = { 0 };//���⻯��ĻҶ�ֵ
//    //�����ۼ��ܶ�
//    gray_distribution[0] = gray_rate[0];
//    for (int i = 1; i < 256; i++) {
//        gray_distribution[i] = gray_distribution[i - 1] + gray_rate[i];
//    }
//    //���¼�����⻯��ĻҶ�ֵ���������롣��ʽ=��L-1��*T������任һ�¹�ʽ�任Ϊ(L-1)��
//    for (int i = 0; i < 256; i++) {
//        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
//    }
//
//    //ֱ��ͼ���⻯������ԭͼÿ�����ص��ֵ
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
//    //�洢��ɫֱ��ͼ��ͼ��ͨ������
//    Mat colorImage;
//    vector<Mat> BGR_plane;
//    //����BGRͨ��
//    split(img, BGR_plane);
//    //�ֱ��BGRͨ������ֱ��ͼ���⻯
//    for (int i = 0; i < BGR_plane.size(); i++) {
//        BGR_plane[i]=equalizeHist(BGR_plane[i]);
//    }
//    //�ϲ�ͨ��
//    merge(BGR_plane, colorImage);
//    //imshow("col", colorImage);
//    //colorImage = showHist(colorImage);
//    return colorImage;
//   
//}
//
//int main(int argc, char* argv[]) {
//    // ͼƬ·��
//    Mat img = imread("D://1.jpg");
//    if (img.empty()) {
//        printf("could not load image...\n");
//        return -1;
//    }
//    imshow("img", img);
//    Mat gray, result1,result2, result3,eqImage,eqcolor;
//    cvtColor(img, gray, COLOR_BGR2GRAY);//��תΪ�Ҷ�ͼ
//    result1 = showHist(gray);//����Ҷ�ֱ��ͼ
//    imshow("gray", gray);
//    imshow("gray_hist", result1);
//
//    eqImage = equalizeHist(gray);//ֱ��ͼ���⻯
//    result2 = showHist(eqImage);
//    imshow("eqImage", eqImage);
//    imshow("result2", result2);
//
//    eqcolor = colorHist(img);//��ɫֱ��ͼ
//    result3 = showHist(eqcolor);
//    imshow("eqcolor", eqcolor);
//    imshow("result3", result3);
//    waitKey(0);
//    return 0;
//}
