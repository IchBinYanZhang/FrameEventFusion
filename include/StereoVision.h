#ifndef STEREOVISION_H
#define STEREOVISION_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include "FrameStream.h"

class StereoVision
{
    public:
        StereoVision();
        StereoVision(FrameStream& s1, FrameStream& s2);
        void SetFrameStream (FrameStream& s1, FrameStream& s2);
        void SetCamCalibration(std::vector<cv::Mat>& intrisicMat, std::vector<cv::Mat>& distCoeff,
                               std::vector<cv::Mat>& rotationMat, std::vector<cv::Mat>& transVec );

        void ImageRectification();
        void DepthEstimate();
        void Tracking();
        void StereoShow(bool is_rectified=false);
        void IsSynchronizedTwoStreams();

        virtual ~StereoVision();
    protected:
    private:
        /// frame streams and videos
        std::vector<cv::VideoCapture> _stream;
        std::vector<std::vector<uint64_t>> _ts;

        /// parameters of all cams
        std::vector<cv::Mat> _intrisic_mat;
        std::vector<cv::Mat> _dist_coeff;
        std::vector<cv::Mat> _rotation_mat;
        std::vector<cv::Mat> _trans_vec;
        int _nx,_ny;






};

#endif // STEREOVISION_H
