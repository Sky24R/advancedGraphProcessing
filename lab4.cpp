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

// 生成高斯噪声（其实就是服从高斯分布的随机数）
double generateGaussianNoise(double mean, double sigma) {
    static double V1, V2, S;
    static int phase = 0;
    double X;
    double U1, U2;
    if (phase == 0) {
        do {
            U1 = (double)rand() / RAND_MAX;
            U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    }
    else {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return mean + sigma * X;
}

// 1、添加高斯噪声
Mat addGaussNoise(Mat src_image, double mean = 0.0, double sigma = 16.0, int k = 2) {
    Mat outImage;
    Mat image = src_image.clone();
    outImage.create(image.rows, image.cols, image.type());
    int PixcelMax = 255, PixcelMin = 0;
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            double temp = image.at<uchar>(x, y) + k * generateGaussianNoise(mean, sigma);
            if (temp > PixcelMax)
                temp = PixcelMax;
            else if (temp < PixcelMin)
                temp = PixcelMin;
            outImage.at<uchar>(x, y) = temp;
        }
    }
    return outImage;
}


// 2、添加椒盐噪声(根据图像的信噪比，添加椒盐噪声)根据选择，可以添加：盐噪声（又称白噪声，白色点255）、胡椒噪声（黑色点）、椒盐噪声。
Mat addPepperSaltNoise(Mat src_image, int addsalt = 0,double SNR = 0.99) {
    Mat outImage;
    Mat image = src_image.clone();
    //outImage.create(image.rows,image.cols,image.type());
    int SP = image.rows * image.cols;
    int NP = SP * (1 - SNR);
    //cout << "SNR " << SNR << " SP " << SP <<  " NP " << NP << endl;
    outImage = image.clone();
    // 噪声点的总个数 与 信噪比有关
    for (int i = 0; i < NP; i++) {
        // 随机选取图像上的点
        int x = (int)(double(rand()) * 1.0 / RAND_MAX * (double)(image.rows-1));
        int y = (int)(double(rand()) * 1.0 / RAND_MAX * (double)(image.cols-1));
        // 峰值（255）和零值（0）出现的概率相同
        if (addsalt==0)
            outImage.at<uchar>(x, y) = 0;
        else if (addsalt==1)
            outImage.at<uchar>(x, y) = 255;
        else {
            int r = rand() % 2;
            outImage.at<uchar>(x, y) = (r > 0) ? 255 : 0;
        }
    }
    return outImage;
}

Mat caddNoise(Mat src_image, int select = 0, double SNR = 0.99, double mean = 0.0, double sigma = 10.0, int k = 2) {
    Mat outImage;
    vector<Mat>  ch;
    Mat image = src_image.clone();
    split(image, ch);
    //cout << "    nn snr  " << SNR << endl;
    for (int k = 0; k < 3; k++) {
        if (select == 0)
            ch.at(k) = addGaussNoise(ch.at(k), mean, sigma, k);
        else if (select == 1)
            ch.at(k) = addPepperSaltNoise(ch.at(k),  select - 1,SNR);
        else if (select == 2)
            ch.at(k) = addPepperSaltNoise(ch.at(k), select - 1, SNR);
        else if (select == 3)
            ch.at(k) = addPepperSaltNoise(ch.at(k), select - 1, SNR);
    }
    cv::merge(ch, outImage);
    return outImage;
}
int ariMeanValueConv(Mat image_block, int size = 5) {
    double** templateMatrix = new double* [size];
    for (int i = 0; i < size; i++) {
        templateMatrix[i] = new double[size];
    }
    int n = size * size;
    int origin = size / 2;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            templateMatrix[i][j] = 1.0;
        }
    }
    double sum = 0;
    for (int k = 0; k < size; k++) {
        for (int m = 0; m < size; m++) {
            sum += (double)templateMatrix[k][m] * image_block.at<uchar>(k, m);

        }
    }
    return (sum / n) + 0.5;

}

//  几何平均
int geometryMeanValueConv(Mat image_block, int size = 5) {
    vector<double> product(5, 1);
    double n = 1.0 / double(size * size);
    for (int k1 = 0; k1 < size; k1++) {
        double temp = 1;
        for (int k2 = 0; k2 < size; k2++) {
            temp = temp * image_block.at<uchar>(k1, k2) ;
            product[k1] = pow(temp, n);
            
        }
    }
    
    return int(product[0] * product[1] * product[2] * product[3] * product[4]);
}

// 谐波均值
int harmonicMeanValueConv(Mat image_block, int size = 5) {
    vector<double> product(5, 1);
    double sum = 0;
    double n = double(size * size);
    for (int k1 = 0; k1 < size; k1++) {
        for (int k2 = 0; k2 < size; k2++) {
            double temp;
            
            temp = 1.0 / double(image_block.at<uchar>(k1, k2));
            
            sum += temp;
        }
    }
    return int(n / sum);
}

// 逆谐波均值
int inverseHarmonicMeanValueConv(Mat image_block, int size = 5) {
    double Q = 1;
    double sum1 = 0, sum2 = 0;
    for (int k1 = 0; k1 < size; k1++) {
        for (int k2 = 0; k2 < size; k2++) {
            sum1 += pow(double(image_block.at<uchar>(k1, k2)), Q);
            sum2 += pow(double(image_block.at<uchar>(k1, k2)), Q + 1);
        }
    }
    return int(sum2 / sum1);
}

// 交换
void exchange(vector<int>& nums, int a, int b) {
    int c = nums[a];
    nums[a] = nums[b];
    nums[b] = c;
    return;
}
//  快速查找中位数
int partition(vector<int>& nums, int begin, int end) {
    int i = begin, j = end + 1;
    int x = nums[begin];
    while (true) {
        while (nums[++i] < x) {// 向右扫描
            if (i == end)
                break;
        }
        while (nums[--j] > x) {// 向左扫描
            if (j == begin)
                break;
        }
        if (i >= j) // 指针相遇，切分位置确定
            break;
        exchange(nums, i, j);// 交换左右逆序元素
    }
    // 运行到最后：i指向从左往右第一个大于x的元素，j指向从右往左第一个小于x的元素。
    exchange(nums, begin, j);// 将切分元素放在切分位置
    return j;
}

// 快速查找中位数
int findMiddleNum(vector<int>& nums, int begin, int end, int n) {
    int i = partition(nums, begin, end);
    if (i == n)
        return nums[i];
    else if (i > n)
        return findMiddleNum(nums, begin, i - 1, n);
    else
        return findMiddleNum(nums, i + 1, end, n);
}

//  中值滤波
int middleValueConv(Mat image_block, int size = 5) {
    int min = 0, k = 0, pos = size * size / 2;
    vector<int> nums;
    for (int k1 = 0; k1 < size; k1++)
        for (int k2 = 0; k2 < size; k2++)
            nums.push_back(image_block.at<uchar>(k1, k2));
    int middle = findMiddleNum(nums, 0, size * size - 1, pos);
    return middle;
}

//  计算图像块的平均值
int computerMeanValue(Mat& image_block, int size = 7) {
    int sum = 0, n = size * size;
    for (int k1 = 0; k1 < size; k1++)
        for (int k2 = 0; k2 < size; k2++)
            sum += image_block.at<uchar>(k1, k2);
    return (sum / n);
}

// 计算图像块的方差
int computerVariance(cv::Mat& image_block, int mean, int size = 7) {
    int sum = 0, n = size * size;
    for (int k1 = 0; k1 < size; k1++)
        for (int k2 = 0; k2 < size; k2++)
            sum += pow((image_block.at<uchar>(k1, k2) - mean), 2);
    return (sum / n);
}

//自适应均值滤波
int selfAdaptionMeanValueConv(Mat image_block, int value, int size = 7 ) {
    int sigma_n = 3000;
    int center = size / 2;
    int mean = computerMeanValue(image_block);
    int sigma = computerVariance(image_block, mean);
    double rate = double(sigma_n) / double(sigma);
    if (rate > 1.0)
        rate = 1;
    int ans = value - rate * (value - mean);
    return ans;
}



// 自适应中值滤波
int selfAdaptionMiddleValueConv(Mat image_, int size = 3 ) {
    int sizemax = 7;
    int center = sizemax / 2;
    int zxy = image_.at<uchar>(center, center);
    int m = size / 2;
    cv::Mat image_block = image_(cv::Rect(center - m, center - m, size, size)).clone();
    std::vector<int> nums;
    for (int k1 = 0; k1 < size; k1++)
        for (int k2 = 0; k2 < size; k2++)
            nums.push_back(image_block.at<uchar>(k1, k2));

    sort(nums.begin(), nums.end());
    int zmin = nums[0], zmax = nums[nums.size() - 1], zmid = nums[(size * size) / 2];
    int A1 = (zmid - zmin), A2 = (zmid - zmax);
    if (A1 > 0 && A2 < 0) {
        int B1 = (zxy - zmin), B2 = (zxy - zmax);
        if (B1 > 0 && B2 < 0)
            return zxy;
        else
            return zmid;
    }
    else {
        if (size < sizemax)
            return selfAdaptionMiddleValueConv(image_, size + 2);
        else
            return zmid;
    }
}


Mat filter(Mat src,int ProcessingMethod(Mat sub_matrix,int ksize),int ksize=5) 
{
        assert(src.channels() || src.channels() == 3);
        Mat image = src.clone();//复制原图
        Mat dst = src.clone();//保存边界扩充
        int border = ksize / 2;
        //边界处理，这里直接调用函数进行
        copyMakeBorder(image, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
        int channels = image.channels();
        
        vector<Mat> BGRchannels,bgr;
        Mat bChannel, gChannel, rChannel;
        if (channels == 3) {
            split(dst, BGRchannels);
            split(image, bgr);
        }
        
        //开始利用模版进行平滑处理
        for (int i = 0; i < src.rows ; i++) {
            for (int j = 0; j < src.cols; j++) {
                double sum[3] = { 0 };
                Mat sub_matrix = dst(Rect(j ,i , ksize, ksize));
                
                
                //灰度
                if (channels == 1) {
                      sum[0] = ProcessingMethod(sub_matrix, ksize);
                }
                //彩色
                else if (channels == 3)
                    {
                       
                        Mat sub_b = BGRchannels[0](Rect(j, i , ksize, ksize));
                        Mat sub_g = BGRchannels[1](Rect(j , i , ksize, ksize));
                        Mat sub_r = BGRchannels[2](Rect(j, i , ksize, ksize));

                        sum[0] = ProcessingMethod(sub_b, ksize);
                        sum[1] = ProcessingMethod(sub_g, ksize);
                        sum[2] = ProcessingMethod(sub_r, ksize);
                    }

                //限定像素值在0-255之间
                for (int i = 0; i < channels; i++) {
                    if (sum[i] < 0)
                        sum[i] = 0;
                    else if (sum[i] > 255)
                        sum[i] = 255;
                }
                if (channels == 1) {
                    image.at<uchar>(i, j ) = static_cast<uchar>(sum[0]);
                }
                else if (channels == 3) {
                    bgr[0].at<uchar>(i , j ) = static_cast<uchar>(sum[0]);
                    bgr[1].at<uchar>(i , j ) = static_cast<uchar>(sum[1]);
                    bgr[2].at<uchar>(i , j) = static_cast<uchar>(sum[2]);
                }
               
            }
            
        }
        if (channels == 3) {
            merge(bgr, image);
        }
        
        return image;
 }

Mat multfilter(Mat src, int ProcessingMethod(Mat sub_matrix, int val,int ksize), int ksize = 5)
{
    assert(src.channels() || src.channels() == 3);
    Mat image = src.clone();//复制原图
    Mat dst = src.clone();//保存边界扩充
    int border = ksize / 2;
    //边界处理，这里直接调用函数进行
    copyMakeBorder(image, dst, border, border, border, border, BorderTypes::BORDER_REFLECT);
    int channels = image.channels();
    
    vector<Mat> BGRchannels,bgr;
    Mat bChannel, gChannel, rChannel;
    if (channels == 3) {

        split(dst, BGRchannels);
        split(image, bgr);
    }

    //开始利用模版进行平滑处理
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double sum[3] = { 0 };
            Mat sub_matrix = dst(Rect(j, i , ksize, ksize));


            //灰度
            if (channels == 1) {
                sum[0] = ProcessingMethod(sub_matrix, image.at<uchar>(i, j ), ksize);
            }
            //彩色
            else if (channels == 3)
            {

                Mat sub_b = BGRchannels[0](Rect(j , i , ksize, ksize));
                Mat sub_g = BGRchannels[1](Rect(j, i , ksize, ksize));
                Mat sub_r = BGRchannels[2](Rect(j, i, ksize, ksize));

                sum[0] = ProcessingMethod(sub_b, bgr[0].at<uchar>(i, j ), ksize);
                sum[1] = ProcessingMethod(sub_g, bgr[1].at<uchar>(i, j ),ksize);
                sum[2] = ProcessingMethod(sub_r, bgr[2].at<uchar>(i, j ), ksize);
            }

            //限定像素值在0-255之间
            for (int i = 0; i < channels; i++) {
                if (sum[i] < 0)
                    sum[i] = 0;
                else if (sum[i] > 255)
                    sum[i] = 255;
            }
            if (channels == 1) {
                image.at<uchar>(i , j ) = static_cast<uchar>(sum[0]);
            }
            else if (channels == 3) {
                bgr[0].at<uchar>(i, j) = static_cast<uchar>(sum[0]);
                bgr[1].at<uchar>(i , j ) = static_cast<uchar>(sum[1]);
                bgr[2].at<uchar>(i , j) = static_cast<uchar>(sum[2]);
                cout << "i " << i << "j " << endl;
            }

        }

    }

    if (channels == 3) {
        merge(bgr, image);
    }

    return image;
}



int main() {
    Mat img = imread("D://1.jpg");
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    /*imshow("gray", gray);
    waitKey(1000);*/
    Mat guass = addGaussNoise(gray);
    /*imshow("guass", guass);
    waitKey(1000);*/
    Mat pepper = addPepperSaltNoise(gray,0);
    /*imshow("pepper", pepper);
    waitKey(1000);*/
    Mat salt = addPepperSaltNoise(gray, 1);
    /*imshow("salt", salt);
    waitKey(1000);*/
    Mat peppersalt = addPepperSaltNoise(gray,2);
    /*imshow("peppersalt", peppersalt);
    waitKey(1000);*/

    //去噪
    /*Mat mean = filter(peppersalt, ariMeanValueConv, 5);
    imshow("mean", mean);
    waitKey(1000);

    Mat geome = filter(peppersalt, geometryMeanValueConv, 5);
    imshow("geome", geome);
    waitKey(1000);

    Mat harmo = filter(peppersalt, harmonicMeanValueConv, 5);
    imshow("harmo", harmo);
    waitKey(1000);

    Mat inharmo = filter(peppersalt, inverseHarmonicMeanValueConv, 5);
    imshow("inharmo", inharmo);
    waitKey(1000);*/


    //Mat middle1 = filter(guass, middleValueConv, 9);//5*5 9*9  
    //imshow("middle1", middle1);
    //waitKey(1000);
    //
    //Mat middle2 = filter(pepper, middleValueConv, 9);//5*5 9*9  
    //imshow("middle2", middle2);
    //waitKey(1000);

    //Mat middle3 = filter(salt, middleValueConv, 9);//5*5 9*9  
    //imshow("middle3", middle3);
    //waitKey(1000);

    //Mat middle4 = filter(peppersalt, middleValueConv, 9);//5*5 9*9  
    //imshow("middle4", middle4);
    //waitKey(1000);


    //Mat selfmean1 = multfilter(guass, selfAdaptionMeanValueConv, 7);//
    //imshow("selfmean1", selfmean1);
    //waitKey(1000);

    //Mat selfmean2 = multfilter(pepper, selfAdaptionMeanValueConv, 7);//
    //imshow("selfmean2", selfmean2);
    //waitKey(1000);

    //Mat selfmean3 = multfilter(salt, selfAdaptionMeanValueConv, 7);//
    //imshow("selfmean3", selfmean3);
    //waitKey(1000);

    //Mat selfmean4 = multfilter(peppersalt, selfAdaptionMeanValueConv, 7);//
    //imshow("selfmean4", selfmean4);
    //waitKey(1000);

    //Mat selfmid1 = filter(guass, selfAdaptionMiddleValueConv, 7);//
    //imshow("selfmid1", selfmid1);
    //waitKey(1000);

    //Mat selfmid2 = filter(pepper, selfAdaptionMiddleValueConv, 7);//
    //imshow("selfmid2", selfmid2);
    //waitKey(1000);

    //Mat selfmid3 = filter(salt, selfAdaptionMiddleValueConv, 7);//
    //imshow("selfmid3", selfmid3);
    //waitKey(1000);

    //Mat selfmid4 = filter(peppersalt, selfAdaptionMiddleValueConv, 7);//
    //imshow("selfmid4", selfmid4);
    //waitKey(1000);



    ////彩色图像
    Mat cguass = caddNoise(img,0);
    /*imshow("cguass", cguass);
    waitKey(1000);*/
    Mat cpeppersalt = caddNoise(img, 3);
    /*imshow("cpeppersalt", cpeppersalt);
    waitKey(1000);*/
    Mat csalt = caddNoise(img, 2);
    /*imshow("csalt", csalt);
    waitKey(1000);*/
    Mat cpepper = caddNoise(img, 1);
    /*imshow("cpepper", cpepper);
    waitKey(1000);*/


    Mat cmean = filter(cpeppersalt, ariMeanValueConv, 5);
    imshow("cmean", cmean);
    waitKey(1000);

    Mat cgeome = filter(cpeppersalt, geometryMeanValueConv, 5);
    imshow("cgeome", cgeome);
    waitKey(1000);
    
    

    return 0;
}