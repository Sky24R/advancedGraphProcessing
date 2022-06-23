#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
using namespace cv;

//����ģ�����ƽ������
Mat Filter(const Mat& src, Mat& dst, int ksize, double** templateMatrix) {
    assert(src.channels() || src.channels() == 3);
    int border = ksize / 2;
    cout << "dst.cols " << dst.cols << " dst.rows " << dst.rows << endl;
    //�߽紦������ֱ�ӵ��ú�������
    copyMakeBorder(src, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = src.channels();
    int cols = dst.cols - border;
    int rows = dst.rows - border;

    cout << "rows " << rows << " cols" << cols << endl;

    //��ʼ����ģ�����ƽ������
    for (int i = border; i < rows; i++) {
        for (int j = border; j < cols; j++) {
            double sum[3] = { 0 };
            for (int k = -border; k <= border; k++) {
                for (int m = -border; m <= border; m++) {
                    //�Ҷ�
                    if (channels == 1) {
                        /*cout << "k + border " << k + border << " m + border " << m + border << endl;
                        cout << "i + k " << i + k << " j + m " << j + m << endl;*/

                        sum[0] += (double)templateMatrix[k + border][m + border] * dst.at<uchar>(i + k, j + m);
                    }
                    //��ɫ
                    else if (channels == 3)
                    {
                        Vec3b rgb = dst.at<Vec3b>(i + k, j + m);
                        auto tmp = templateMatrix[border + k][border + m];
                        sum[0] += tmp * rgb[0];
                        sum[1] += tmp * rgb[1];
                        sum[2] += tmp * rgb[2];
                    }
                }
            }
            //�޶�����ֵ��0-255֮��
            for (int i = 0; i < channels; i++) {
                if (sum[i] < 0)
                    sum[i] = 0;
                else if (sum[i] > 255)
                    sum[i] = 255;
            }
            if (channels == 1) {
                dst.at<uchar>(i, j) = static_cast<uchar>(sum[0]);
            }
            else if (channels == 3) {
                Vec3b rgb;
                rgb[0] = static_cast<uchar>(sum[0]);
                rgb[1] = static_cast<uchar>(sum[1]);
                rgb[2] = static_cast<uchar>(sum[2]);

                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    return dst;
}
//���þ�ֵģ��ƽ���Ҷ�ͼ��
void MeanFilter(const Mat& src, Mat& dst, int ksize) {
    //����һ����ά����
    double** templateMatrix = new double* [ksize];
    for (int i = 0; i < ksize; i++) {
        templateMatrix[i] = new double[ksize];
    }
    int tmp = ksize * ksize;
    int origin = ksize / 2;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            //ÿ��λ����ֱ�������ƽ��ֵ
            templateMatrix[i][j] = 1.0 / tmp;
        }
    }
    dst = Filter(src, dst, ksize, templateMatrix);

}

//���ø�˹ģ��ƽ���Ҷ�ͼ��
void GaussianFilter(const Mat& src, Mat& dst, int ksize, double sigma) {
    const static double pi = 3.1415926;
    //���ݴ��ڴ�С��sigma���ɸ�˹�˲�ģ�棬������һ����ά���飬������ɵĸ�˹ģ�����
    double** templateMatrix = new double* [ksize];
    for (int i = 0; i < ksize; i++)
        templateMatrix[i] = new double[ksize];
    int origin = ksize / 2; //��ģ������Ϊԭ��
    double x2, y2;
    double sum = 0;
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - origin, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(double(j - origin), 2);
            //��˹����ǰ�ĳ������Բ��ü��㣬���ڹ�һ���Ĺ�������ȥ
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            sum += g;
            templateMatrix[i][j] = g;
        }
    }
    double k = 1 / sum;
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            templateMatrix[i][j] *= k;
        }
    }
    dst = Filter(src, dst, ksize, templateMatrix);
    
}


//���� Laplacian��Robert��Sobel ģ���񻯻Ҷ�ͼ��
int FilterProcessing(Mat src, Mat &dst, Mat filter, double ProcessingMethod(Mat filterArea, Mat filter))
{
    Mat src_padding = src.clone();
    Mat filterArea;
    int padding = (filter.rows - 1) / 2;
    //padding the border
    copyMakeBorder(src, src_padding, padding, padding, padding, padding, BORDER_REPLICATE);

    if (dst.type() == CV_8U)
    {
        for (int y = padding; y < src_padding.rows - padding; y++)
        {
            for (int x = padding; x < src_padding.cols - padding; x++)
            {
                filterArea = src_padding(Range(y - padding, y + padding + 1), Range(x - padding, x + padding + 1));
                dst.at<uchar>(y - padding, x - padding) = cvRound(ProcessingMethod(filterArea, filter));
            }
        }
    }
    else if (dst.type() == CV_64F)
    {
        for (int y = padding; y < src_padding.rows - padding; y++)
        {
            for (int x = padding; x < src_padding.cols - padding; x++)
            {
                filterArea = src_padding(Range(y - padding, y + padding + 1), Range(x - padding, x + padding + 1));
                dst.at<double>(y - padding, x - padding) = ProcessingMethod(filterArea, filter);
            }
        }
    }
    else
    {
        cout << "type error" << endl;
    }

    return 0;
}
double LinearFilterCalc(Mat filterArea, Mat linearFilter)
{
    double result = 0;
    for (int y = 0; y < filterArea.rows; y++)
    {
        for (int x = 0; x < filterArea.cols; x++)
        {
            result += (double(filterArea.at<uchar>(y, x))) * (linearFilter.at<double>(y, x));
        }
    }
    return result;
}
int LaplacianFilterProcessing(Mat src, Mat &dst, Mat laplacianFilter, Mat laplacianFilterImg, double c)
{

    FilterProcessing(src, laplacianFilterImg, laplacianFilter, LinearFilterCalc);

    //���㲢�궨�񻯽����ֱ��saturate_cast<uchar>
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            dst.at<uchar>(y, x) = saturate_cast<uchar>(src.at<uchar>(y, x) + cvRound(c * laplacianFilterImg.at<double>(y, x)));
        }
    }

    return 0;
}
int LaplacianSharpen(Mat src, Mat &dst,  double c, int filterNum)//dst���񻯺�Ľ��ͼ��
{
    Mat laplacianFilter_n4 = (Mat_<double>(3, 3) <<
        0, 1, 0,
        1, -4, 1,
        0, 1, 0);
    /*Mat laplacianFilter_n8 = (Mat_<double>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);*/
    Mat laplacianFilter;
    laplacianFilter = laplacianFilter_n4;
   

    Mat laplacianFilterImg = Mat::zeros(src.size(), CV_64F);//�˲���õ��ı�Եͼ��

    LaplacianFilterProcessing(src, dst, laplacianFilter, laplacianFilterImg, c);

    
    return 0;
}
Mat laplacia_rbgfilter(const Mat& image, Mat& gray) {
    vector<Mat> BGRchannels;
    split(image, BGRchannels);
    Mat bChannel, gChannel, rChannel;
    bChannel = BGRchannels.at(0);
    gChannel = BGRchannels.at(1);
    rChannel = BGRchannels.at(2);

    vector<Mat> BGRchannels_merge(3);

    /*Laplacian*/
    int lablacianFilterNum = 1;//1Ϊ����Ϊ-4��laplacianģ�壬2Ϊ����Ϊ-8��
    double cLap = -1;
    Mat laplacianSharpen_bChannel = Mat::zeros(gray.size(), gray.type());
    Mat laplacianSharpen_gChannel = Mat::zeros(gray.size(), gray.type());
    Mat laplacianSharpen_rChannel = Mat::zeros(gray.size(), gray.type());
    Mat laplacianSharpenImg = Mat::zeros(image.size(), image.type());

    LaplacianSharpen(bChannel, laplacianSharpen_bChannel, cLap, lablacianFilterNum);
    LaplacianSharpen(gChannel, laplacianSharpen_gChannel, cLap, lablacianFilterNum);
    LaplacianSharpen(rChannel, laplacianSharpen_rChannel, cLap, lablacianFilterNum);

    BGRchannels_merge.at(0) = laplacianSharpen_bChannel;
    BGRchannels_merge.at(1) = laplacianSharpen_gChannel;
    BGRchannels_merge.at(2) = laplacianSharpen_rChannel;
    merge(BGRchannels_merge, laplacianSharpenImg);

    return laplacianSharpenImg;

}
//ʵ������ʹ��Robert��
//�Ҷ��£�
int robert_Sharpen(Mat src, Mat &dst,  double c)
{
    Mat robertFilterImg = Mat::zeros(src.size(), CV_64F);//�˲���õ��ı�Եͼ��
    for (int y = 0; y < src.rows - 1; y++)
    {
        for (int x = 0; x < src.cols - 1; x++)
        {
            robertFilterImg.at<double>(y, x) =
                abs(src.at<uchar>(y + 1, x + 1) - src.at<uchar>(y, x)) + abs(src.at<uchar>(y + 1, x) - src.at<uchar>(y, x + 1));
            dst.at <uchar>(y, x) = saturate_cast<uchar>(src.at<uchar>(y, x) + cvRound(c * robertFilterImg.at<double>(y, x)));
        }
    }
   
    return 0;
}
//��ɫ�£�
Mat robot_rbgfilter(const Mat& image, Mat& gray) {
    vector<Mat> BGRchannels;
    split(image, BGRchannels);
    Mat bChannel, gChannel, rChannel;
    bChannel = BGRchannels.at(0);
    gChannel = BGRchannels.at(1);
    rChannel = BGRchannels.at(2);
    vector<Mat> BGRchannels_merge(3);
    /*Robert*/
    double cRob = 1;
    Mat robertSharpen_bChannel = Mat::zeros(gray.size(), gray.type());
    Mat robertSharpen_gChannel = Mat::zeros(gray.size(), gray.type());
    Mat robertSharpen_rChannel = Mat::zeros(gray.size(), gray.type());
    Mat robertSharpenImg = Mat::zeros(image.size(), image.type());

    robert_Sharpen(bChannel, robertSharpen_bChannel, cRob);
    robert_Sharpen(gChannel, robertSharpen_gChannel, cRob);
    robert_Sharpen(rChannel, robertSharpen_rChannel, cRob);

    BGRchannels_merge.at(0) = robertSharpen_bChannel;
    BGRchannels_merge.at(1) = robertSharpen_gChannel;
    BGRchannels_merge.at(2) = robertSharpen_rChannel;
    merge(BGRchannels_merge, robertSharpenImg);

    return robertSharpenImg;
}
//ʵ������sobel��
int SobelSharpen(Mat src, Mat& dst,  double c)
{
    Mat sobelFilterImg = Mat::zeros(src.size(), CV_64F);//�˲���õ��ı�Եͼ��
    Mat filterArea;
    Mat sobelFilter_x = (Mat_<double>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);
    Mat sobelFilter_y = (Mat_<double>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);
    for (int y = 1; y < src.rows - 1; y++)
    {
        for (int x = 1; x < src.cols - 1; x++)
        {
            filterArea = src(Range(y - 1, y + 1 + 1), Range(x - 1, x + 1 + 1));
            sobelFilterImg.at<double>(y, x) = abs(LinearFilterCalc(filterArea, sobelFilter_x)) + abs(LinearFilterCalc(filterArea, sobelFilter_y));
            dst.at <uchar>(y, x) = saturate_cast<uchar>(src.at<uchar>(y, x) + cvRound(c * sobelFilterImg.at<double>(y, x)));
        }
    }
    
    return 0;
}
//Sobel��ɫ��
Mat sobel_rbgfilter(const Mat& image, Mat& gray) {
    vector<Mat> BGRchannels;
    split(image, BGRchannels);
    Mat bChannel, gChannel, rChannel;
    bChannel = BGRchannels.at(0);
    gChannel = BGRchannels.at(1);
    rChannel = BGRchannels.at(2);

    vector<Mat> BGRchannels_merge(3);
    /*Sobel*/
    double cSob = 0.5;
    Mat sobelSharpen_bChannel = Mat::zeros(gray.size(), gray.type());
    Mat sobelSharpen_gChannel = Mat::zeros(gray.size(), gray.type());
    Mat sobelSharpen_rChannel = Mat::zeros(gray.size(), gray.type());
    Mat sobelSharpenImg = Mat::zeros(image.size(), image.type());

    SobelSharpen(bChannel, sobelSharpen_bChannel, cSob);
    SobelSharpen(gChannel, sobelSharpen_gChannel, cSob);
    SobelSharpen(rChannel, sobelSharpen_rChannel, cSob);

    BGRchannels_merge.at(0) = sobelSharpen_bChannel;
    BGRchannels_merge.at(1) = sobelSharpen_gChannel;
    BGRchannels_merge.at(2) = sobelSharpen_rChannel;
    merge(BGRchannels_merge, sobelSharpenImg);
    return sobelSharpenImg;

}

//ʵ���ģ����ø������˲��㷨��ǿ�Ҷ�ͼ��
Mat HighboostFilterProcessing(Mat& gray)
{
    double k = 1.5;//k>1ʱΪ�������˲�

    Mat meanFilter_3x3 = ((double)1 / 9) * Mat::ones(3, 3, CV_64F);//CV_64F��Ӧdouble����CV_32F����double�ᱨ��
    Mat filter;
    filter = meanFilter_3x3;
    Mat highboostFilterImg = Mat::zeros(gray.size(), gray.type());
    Mat unsharpMask = Mat::zeros(gray.size(), CV_64F);//�����ڱ�
    Mat blurImg = Mat::zeros(gray.size(), gray.type());

    FilterProcessing(gray, blurImg, filter, LinearFilterCalc);

    gray.convertTo(gray, CV_64F);
    blurImg.convertTo(blurImg, CV_64F);
    unsharpMask = gray - blurImg;//��ԭͼ���м�ȥģ��ͼ�񣬲����Ĳ�ֵͼ���Ϊģ��
    gray.convertTo(gray, CV_8U);

    for (int y = 0; y < highboostFilterImg.rows; y++)
    {
        for (int x = 0; x < highboostFilterImg.cols; x++)
        {
            //��k=1ʱ�����ǵõ����涨��ķ����ڱΡ���k>1ʱ���ô����Ϊ�������˲�����k<1ʱ����ǿ������ģ��Ĺ��ס�
            highboostFilterImg.at <uchar>(y, x) = saturate_cast<uchar>(gray.at<uchar>(y, x) + k * unsharpMask.at<double>(y, x));
        }
    }

    
    return highboostFilterImg;
}

//������ʵ��
int main(int argc, char* argv[]) {
    /*Mat img = imread("D://1.jpg");
    Mat grayImage;
    
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    imshow("grayImage", grayImage);*/
    const char* imageName = "D://1.jpg";
    Mat img, grayImage;
    //�����ɫͼƬ
    img = imread(imageName);
    //����Ҷ�ͼƬ
    grayImage = imread(imageName, 0);
   
    Mat dst1 = grayImage.clone();
    Mat dst2 = grayImage.clone();
    Mat dst3 = grayImage.clone();
    Mat dst4 = grayImage.clone();
    Mat dst5 = grayImage.clone();
    Mat dst6 = grayImage.clone();
    Mat dst7 = grayImage.clone();

    MeanFilter(grayImage, dst1, 9);
    imshow("MeanFilter", dst1);
    waitKey(1000);
    GaussianFilter(grayImage, dst2, 9, 2);
    imshow("GaussianFilter", dst2);
    waitKey(1000);
    LaplacianSharpen(grayImage, dst3, 1, 1);
    imshow("LaplacianSharpen", dst3);
    waitKey(1000);
    robert_Sharpen(grayImage, dst4, 1);
    imshow("robert_Sharpen", dst4);
    waitKey(1000);
    SobelSharpen(grayImage, dst5, 1);
    imshow("SobelSharpen", dst5);
    waitKey(1000);
    Mat Highboost = HighboostFilterProcessing(dst6);
    imshow("HighboostFilter", Highboost);
    waitKey(1000);
    //��ɫͼƬ
    Mat dst8 = grayImage.clone();
    Mat dst9 = grayImage.clone();
    Mat dst10 = grayImage.clone();
    Mat dst11 = grayImage.clone();
    MeanFilter(img, dst7, 9);
    imshow("cMeanFilter", dst7);
    waitKey(1000);
    GaussianFilter(img, dst8, 9, 2);
    imshow("cGaussianFilter", dst8);
    waitKey(1000);
    Mat clap = laplacia_rbgfilter(img, dst9);
    imshow("claplacia_rbgfilter", clap);
    waitKey(1000);
    Mat crobot = robot_rbgfilter(img, dst10);
    imshow("crobot_rbgfilter", crobot);
    waitKey(1000);
    Mat csobel = sobel_rbgfilter(img, dst11);
    imshow("csobel_rbgfilter", csobel);

    waitKey(1000);
    //destroyAllWindows();
    return 0;
}

