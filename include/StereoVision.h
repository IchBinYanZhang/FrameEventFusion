#ifndef STEREOVISION_H
#define STEREOVISION_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <opencv/cv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv/highgui.h>
#include "FrameStream.h"
#include "StipDetector.h"

class StereoVision
{
    public:
        StereoVision();
        StereoVision(FrameStream& s1, FrameStream& s2);
        void SetFrameStream (FrameStream& s1, FrameStream& s2);
        void SetCamCalibration(std::vector<cv::Mat>& intrisicMat, std::vector<cv::Mat>& distCoeff,
                               std::vector<cv::Mat>& rotationMat, std::vector<cv::Mat>& transVec );

        void ImageRectification(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& out1,
                                      cv::Mat& out2);
        void ImageROI(cv::Mat& frame, cv::Mat& roi);
        void DepthEstimate(cv::Mat& frame1, cv::Mat& frame2);
        void Tracking();
        void DepthShow();
        void StereoShow(bool is_rectified=false);
        void IsSynchronizedTwoStreams();

        virtual ~StereoVision();
    protected:
        void Close(const cv::Mat& src, cv::Mat& dst, int kernelsize);
    private:
        /// frame streams and videos
        std::vector<cv::VideoCapture> _stream;
        std::vector<std::vector<uint64_t>> _ts;
        cv::Ptr<cv::BackgroundSubtractorMOG2> _pMOG; // ROI



        /// parameters of all cams
        std::vector<cv::Mat> _intrisic_mat;
        std::vector<cv::Mat> _dist_coeff;
        std::vector<cv::Mat> _rotation_mat;
        std::vector<cv::Mat> _trans_vec;
        int _nx,_ny;

        ///params for image rectification
        cv::Mat _cam1map1, _cam1map2, _cam2map1, _cam2map2;






};

#endif // STEREOVISION_H
