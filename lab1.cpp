#include<opencv2/opencv.hpp>

#include<iostream>
using namespace cv;
using namespace std;
//��ֵ���任
void binarization(Mat& gray, float thresh) {
    Mat result = gray.clone();
    //��ָ��������أ��ٶ����
    uchar* p;
    for (int i = 0; i < result.rows; i++) {
        p = result.ptr<uchar>(i);//��ȡÿ���׵�ַ
        for (int j = 0; j < result.cols; j++) {
            if (p[j] < thresh)
                p[j] = 0;
            else
                p[j] = 255;
        }
    }

    imshow("binary", result);
}

//�����任
void logTransfer(Mat& gray,float c=1) {
    Mat srcImage(gray);
    Mat dstImage(srcImage.size(), srcImage.type());
    //����1+r
    add(gray, Scalar(1.0), srcImage);
    //ת��Ϊ������
    srcImage.convertTo(srcImage, CV_64F);
    //����log(1+r) log:��һ������Ϊ����ͼ�񣬵ڶ�������Ϊ�õ��Ķ���ֵ
    log(srcImage, dstImage);
    //������c=1
    dstImage = c * dstImage;
    //��һ������
    normalize(dstImage, dstImage, 0, 255, NORM_MINMAX);
    convertScaleAbs(dstImage, dstImage);
    imshow("logTransfer", dstImage);
}

//٤��任
   //gammaֵ������ֵ��ͬͼƬ���ֳ���ͬ��Ч��
void gammaTranfer(Mat& gray,double gamma) {
    //gammaֵ������ֵ��ͬͼƬ���ֳ���ͬ��Ч��
    Mat grayImg;
    gray.convertTo(grayImg, CV_64F, 1.0 / 255, 0);
    Mat gammaImg;
    pow(grayImg, gamma, gammaImg);
    gammaImg.convertTo(gammaImg, CV_8U, 255, 0);
    imshow("gammaTranfer", gammaImg);

}

//��ɫͼ��任
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
	// ͼƬ·��
	Mat img = imread("D://4.jpg");
	if (img.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("img", img);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);//��תΪ�Ҷ�ͼ
    imshow("gray", gray);
    binarization(gray, 128);
    logTransfer(gray);//�����任
    double gamma = 1.1;
    gammaTranfer(gray,gamma);//gamma�任
    colorTransfer(img);//��ɫ�任

	waitKey(0);
	return 0;

}
