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
#include "LocalBinaryPattern.h"
#include "MyHistogram.h"

class StereoVision
{
    public:
        enum Tracking2DMethod {TRACKINGBYDETECTION, CAMSHIFT};
        enum Tracking3DMethod {TRACKINGBYDETECTION3, EPICAMSHIFT, KALMAN3D};
        enum FeatureMethod {HSVLBP, HOG};

        StereoVision();
        StereoVision(FrameStream& s1, FrameStream& s2);
        void SetFrameStream (FrameStream& s1, FrameStream& s2);
        void SetCamCalibration(std::vector<cv::Mat>& intrisicMat, std::vector<cv::Mat>& distCoeff,
                               std::vector<cv::Mat>& rotationMat, std::vector<cv::Mat>& transVec );

        void ImageRectification(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& out1,
                                      cv::Mat& out2);
        void ImageROIMoG2(cv::Mat& frame, cv::Mat& roi);
        void ImageROIAll( cv::Mat& frame, cv::Mat& roi);

        void DenseDepthEstimate(cv::Mat& frame1, cv::Mat& frame2);
        bool Tracking3D(const cv::Mat& f0_pre, const cv::Mat& f0_cur, cv::Rect& bd0, cv::Mat& hist0,
                              const cv::Mat& f1_pre, const cv::Mat& f1_cur, cv::Rect& bd1, cv::Mat& hist1,
                              Tracking3DMethod method, FeatureMethod method_feature);

        void Tracking3DKalman(Rect& boundingbox0, Rect& boundingbox1, Mat& pt_in, float vx, float vy, float vz, Mat& pt_out, Mat&, Point2f& p1, Point2f& p2);

        void DepthShow();
        void StereoShow(bool is_rectified=false);
        void IsSynchronizedTwoStreams();
        void BlockMatching(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& frame1_roi,cv::Mat& frame2_roi, cv::Mat& out);
        void BlockMatching2(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& frame1_roi, cv::Mat& out);
        void Tracking3DInitialize( cv::Mat& f1_pre, cv::Mat& f1_cur, cv::Rect& bd1,
                                         cv::Mat& f2_pre, cv::Mat& f2_cur, cv::Rect& bd2,
                                         cv::Mat& center);
        void HomographyToGround(cv::Mat& img_cb, bool);
        bool Tracking2D(const cv::Mat& f1, const cv::Mat& f0, cv::Rect& bd, cv::Mat& hist, Tracking2DMethod method);
        virtual ~StereoVision();


    protected:
        void Close(const cv::Mat& src, cv::Mat& dst, int kernelsize);
        void Open(const cv::Mat& src, cv::Mat& dst, int kernelsize);

        inline float SAD(cv::Mat& set1, cv::Mat& set2);
        inline void MultiBoundingBoxFromROI(cv::Mat& roi, std::vector<cv::Rect>& boundRect);
        inline void SingleBoundingBoxFromROI(cv::Mat& roi, cv::Rect& bd);
        inline void ShowBoundingBox(const cv::Mat&, cv::Rect&);
        inline void ShowBoundingBox(const cv::Mat& drawing, std::vector<cv::Rect>& bd);
        inline void ShowTracking(const cv::Mat& f_current, cv::Rect& bd, std::vector<cv::Point2f>& trajectory, cv::Mat& out);
        inline void ImagePreprocessing(const cv::Mat& f, cv::Mat& out);
        void CLBP( Mat& src, Mat& dst, int radius, int neighbors);
        inline void HistDisplay(const cv::Mat& hist, const int nbins, const float* histRange );
        void PointSetPerspectiveTransform(const std::vector<Point2f>& in, std::vector<Point2f>& out, cv::Mat& H );
        inline void MaskFromRect(const cv::Mat& img, cv::Rect& bd, cv::Mat& out);
        inline void FundamentalMatrixFromCalibration(const cv::Mat& K, const cv::Mat& R, const cv::Mat& T,
                                                           const cv::Mat& K_prime, const cv::Mat& R_prime, const cv::Mat& T_prime,
                                                           cv::Mat& F);

        inline void UpdateGeometryByImageResize(float fx, float fy);

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
        std::vector<cv::Mat> _projection_mat;
        std::vector<cv::Mat> _projection_mat_rec;

        cv::Mat _H; // for the specific cam of the corresponding checkerboard image
        int _nx,_ny;

        ///params for image rectification
        cv::Mat _cam1map1, _cam1map2, _cam2map1, _cam2map2;






};

#endif // STEREOVISION_H
