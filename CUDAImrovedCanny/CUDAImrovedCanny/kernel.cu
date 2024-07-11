//Source code for improved Canny
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

//For CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//For CPP
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

//For openCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp> //for filtering
#include <opencv2/cudafilters.hpp>  //for filtering
#include <opencv2/cudaarithm.hpp> //for abs
#include <opencv2/imgcodecs.hpp>     // Image file reading and writing

//For Thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define mode 1 // 0 for sobel, 1 for less, 2 for more

void createGaussianKernel(int);
void cannyDector();
void useGaussianBlur();
void getGradientImg();
void nonMaxSuppress();
void lessHysteresisThreshold(int, int);
void moreHysteresisThreshold();
cv::Mat combineImage();

//openCV variables
cv::Mat oriImage, bluredImage, edgeMagImage, edgeAngImage, thinEdgeImage, thresholdImage;
cv::Mat lowTho, highTho, sobelX, sobelY;
int* gaussianMask, maskRad, maskWidth = 0, maskSum = 0;
float sigma = 0.0, avgGradient = 0.0, var = 0.0;

//CUDA variables
cv::cuda::GpuMat gpuImg;

int main()
{	
    //Read image
    cv::Mat combinedImage, tempImg;
    oriImage = cv::imread("F:\\Improved-Canny-master\\image\\lena.jpg", 0);

    if (oriImage.empty())
    {
        printf("Image read failed\n");
        exit(-1);
    }
    gpuImg.upload(oriImage); //uploading in GPU
    std::cout << "Image UpLoading Done!" << std::endl;
    
    int channels = oriImage.channels();
    if (channels == 1) {
        std::cout << "The image is a grayscale image." << std::endl;
    }
    if (channels == 3) {
        std::cout << "The image is a color image (BGR)." << std::endl;
    }




	return 0;
}
