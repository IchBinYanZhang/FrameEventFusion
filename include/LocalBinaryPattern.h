#ifndef LOCALBINARYPATTERN_H
#define LOCALBINARYPATTERN_H


#include <cstdio>
#include <vector>
#include <math.h>
#include <iostream>
#include <opencv/cv.hpp>


using namespace std;
using namespace cv;

class LocalBinaryPattern
{
    public:
        LocalBinaryPattern(int radius, int points, bool uniform=true);
        virtual ~LocalBinaryPattern();
        int countSetBits(int code);
        bool checkUniform(int code);
        void UniformLBP( cv::Mat& src, cv::Mat& dst);
        int GetNumBins();
        int GetRanges();
    protected:
        inline int rightshift(int num, int shift);
    private:
        int _r;
        int _n_pts;
        bool _uniform;
};

#endif // LOCALBINARYPATTERN_H
