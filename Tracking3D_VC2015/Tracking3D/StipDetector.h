#ifndef STIPDETECTOR_H
#define STIPDETECTOR_H

#include<iostream>
#include<vector>
#include<highgui.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/optflow.hpp>


using namespace std;
using namespace cv;


/// notice that this class only accept grayvalue images.

class StipDetector
{
    public:
        enum RoiMethod {TemporalThreshold, GM2, SimpleFlow, TVL1Flow, Manual};
        enum ScoreMethod {Harris, MinEigen};
        enum FeatureMethod{STIP, ORB};

        StipDetector();
        StipDetector(const Mat& frame_previous, const Mat& frame_current);
        virtual ~StipDetector();
        void VideoKeypointDisplay( Mat& frame, const vector<KeyPoint>& corners);

        void SetFineScale(float scale);
        void SetLevelNum( int n_level);
        void SetMethodScore(StipDetector::ScoreMethod method);
        void SetMethodROI(StipDetector::RoiMethod method);
        void SetBoundingBox(cv::Rect& bbox);
        void SetFrames(const cv::Mat& f1, const cv::Mat& f2);

        void detect(StipDetector::FeatureMethod method);
        void ClearPoints();
        void DefineROI();
        void GetROI(cv::Mat& fg);
        void GetScore(cv::Mat& score);
        void GetKeyPoints(vector<cv::KeyPoint>& corners);
        void GetDescriptorORB( cv::Mat& out);
        void VideoKeypointDisplay(std::string window_name );

    protected:

        void Gradient (const Mat& src, Mat& gradx, Mat& grady, bool use_sobel = true);
        void MotionTensorScore (const cv::Mat& frame_current, const cv::Mat& frame_previous, cv::Mat& score, float rho);
        void Dilation(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);
        void Open(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);
        void Close(const cv::Mat& src, cv::Mat& dst, int kernelsize=5);



    private:
        int _n_level; // number of scale-space levels (or Pyramid levels)
        double _scale_base; // the finest scale, i.e. size of gaussian kernel
        std::vector<cv::KeyPoint> _corners; // keypoints
        Mat _frame_current; // f2
        Mat _frame_previous; // f1
        RoiMethod _roi_method;
        ScoreMethod _score_method;

        Mat _score, _score_dilate, _score_peak;
        Mat _roi;
        Rect _bounding_box;
        cv::Ptr<cv::BackgroundSubtractorMOG2> _pMOG;


};

#endif // STIPDETECTOR_H
