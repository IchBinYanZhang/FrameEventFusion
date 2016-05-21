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
        void SetFrameStream (std::string& s1, std::string& s2);
        void SetCamCalibration(std::vector<cv::Mat>& intrisicMat, std::vector<std::vector<double>>& distCoeff,
                               std::vector<cv::Mat>& rotationMat, std::vector<std::vector<double>>& transVec );

        void ImageRectification();
        void DepthEstimate();
        void Tracking();
        void StereoShow(bool is_rectified=false);

        virtual ~StereoVision();
    protected:
    private:
        /// frame streams and videos
        std::vector<FrameStream> _stream;


        /// parameters of all cams
        std::vector<cv::Mat> _intrisic_mat;
        std::vector<std::vector<double>> _dist_coeff;
        std::vector<cv::Mat> _rotation_mat;
        std::vector<std::vector<double>> _trans_vec;
        int nx; // image size
        int ny; // image size






};

#endif // STEREOVISION_H
