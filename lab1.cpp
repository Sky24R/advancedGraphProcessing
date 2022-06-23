#include<opencv2/opencv.hpp>

#include<iostream>
using namespace cv;
using namespace std;
//二值化变换
void binarization(Mat& gray, float thresh) {
    Mat result = gray.clone();
    //用指针访问像素，速度最快
    uchar* p;
    for (int i = 0; i < result.rows; i++) {
        p = result.ptr<uchar>(i);//获取每行首地址
        for (int j = 0; j < result.cols; j++) {
            if (p[j] < thresh)
                p[j] = 0;
            else
                p[j] = 255;
        }
    }

    imshow("binary", result);
}

//对数变换
void logTransfer(Mat& gray,float c=1) {
    Mat srcImage(gray);
    Mat dstImage(srcImage.size(), srcImage.type());
    //计算1+r
    add(gray, Scalar(1.0), srcImage);
    //转换为浮点数
    srcImage.convertTo(srcImage, CV_64F);
    //计算log(1+r) log:第一个参数为输入图像，第二个参数为得到的对数值
    log(srcImage, dstImage);
    //这里设c=1
    dstImage = c * dstImage;
    //归一化处理
    normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
    convertScaleAbs(dstImage, dstImage);
    imshow("logTransfer", dstImage);
}

//伽马变换
   //gamma值，随着值不同图片呈现出不同的效果
void gammaTranfer(Mat& gray,double gamma) {
    //gamma值，随着值不同图片呈现出不同的效果
    Mat grayImg;
    gray.convertTo(grayImg, CV_64F, 1.0 / 255, 0);
    Mat gammaImg;
    pow(grayImg, gamma, gammaImg);
    gammaImg.convertTo(gammaImg, CV_8U, 255, 0);
    imshow("gammaTranfer", gammaImg);

}

//彩色图像变换
void colorTransfer(Mat& image) {
    Mat colorpicture;
    image.copyTo(colorpicture);
    //HSV
    Mat hsvImg;
    cvtColor(image, hsvImg, COLOR_BGR2HSV);
    std::vector<Mat> hsv;
    split(hsvImg, hsv);
    hsv[0] = (Scalar::all(180) - hsv[0]);
    merge(hsv, colorpicture);
    imshow("hsv", colorpicture);
    //RGB
    colorpicture = Scalar::all(255) - image;
    imshow("rgb", colorpicture);

}

int main(int argc, char** argv) {
	// 图片路径
	Mat img = imread("D://4.jpg");
	if (img.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("img", img);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);//先转为灰度图
    imshow("gray", gray);
    binarization(gray, 128);
    logTransfer(gray);//对数变换
    double gamma = 1.1;
    gammaTranfer(gray,gamma);//gamma变换
    colorTransfer(img);//补色变换

	waitKey(0);
	return 0;

}
