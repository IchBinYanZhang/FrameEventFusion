#ifndef MYHISTOGRAM_H
#define MYHISTOGRAM_H
#include <cstdio>
#include <vector>
#include <math.h>
#include <iostream>
#include <opencv/cv.hpp>

#include "LocalBinaryPattern.h"

using namespace std;
using namespace cv;

class MyHistogram
{
    public:
        enum FeatureSpace {RGB, HSV, LBP};
        MyHistogram();
        MyHistogram(const cv::Mat& image, const cv::Mat& mask, FeatureSpace method);
        void SetImage(const cv::Mat& image);
        void SetMask(const cv::Mat& mask);
        void GetHist(cv::Mat& out);
        void ComputeHist();
        void BackProjection(const cv::Mat& in, cv::Mat& backproj);




        virtual ~MyHistogram();
    protected:

        inline void SingleBoundingBoxFromROI(cv::Mat& roi, cv::Rect& bd);
    private:
        cv::Mat _img, _mask;
        FeatureSpace _method;
        cv::Mat _hist;










};

#endif // MYHISTOGRAM_H
