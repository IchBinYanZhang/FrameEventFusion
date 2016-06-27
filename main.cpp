#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <opencv/cv.h>
#include "FrameStream.h"
#include "StereoVision.h"
#include "FrameEventFuser.h"


using namespace std;
int main(int argc, char** argv)
{

    /// argument transfer and setup
    if(argc != 6)
    {
        std::cerr<< "./program [filename1] [filename_timestamp1] [filename2] [filename_timestamp2] [traj] " <<std::endl;
        return -1;
    }
    std::string filename_frame= argv[1];
    std::string filename_timestamp = argv[2];
    std::string filename_frame2= argv[3];;
    std::string filename_timestamp2 = argv[4];
    std::string filename_traj = argv[5];

    std:: cout << "read frame stream1" <<std::endl;

    FrameStream frames(filename_frame, filename_timestamp);
    std:: cout << "read frame stream2" <<std::endl;

    FrameStream frames2(filename_frame2, filename_timestamp2);

    StereoVision sv (frames, frames2);
    sv.IsSynchronizedTwoStreams();

    ///>> frame event display
//    EventStream events1;
//
//    events1.ReadFromFile(filename_events, 40000000);
//    FrameEventFuser fuser(frames, events1);
//    fuser.FrameEventShow(false);
    /// <<<


    /// >>> 3D tracking
    /// calibration data
    /// the data is from the Matlab calibration toolbox
    /// the data is specifically for SenseEmotion Experiment 2A (left and middle cam)
    std::vector<cv::Mat> intrisicMat;
    std::vector<cv::Mat> distCoeff;
    std::vector<cv::Mat> rotationMat;
    std::vector<cv::Mat> transVec;
    cv::Mat ground_plane; // 3X2 matrix, Euclidean space
//
////    /// geometry middle-left
////    cv::Mat C = (cv::Mat_<double>(3,3)<< 1405.7, 0, 644.5536, 0, 1402.7, 491.14,0,0,1);
////    intrisicMat.push_back(C);
////    C = (cv::Mat_<double>(3,3)<< 1403.9,0,640.4511,0,1401.3,475.6392,0,0,1);
////    intrisicMat.push_back(C);
////
////    cv::Mat R = (cv::Mat_<double>(3,3)<< 1,0,0,0,1,0,0,0,1);
////    rotationMat.push_back(R);
////    R = (cv::Mat_<double>(3,3)<< 0.9512,1.1875e-4,-0.3084,-0.0964,0.9500,-0.2969,0.2930,0.3122,0.9037);
////    rotationMat.push_back(R);
////
////    cv::Mat dc = (cv::Mat_<double>(4,1) << -0.1557, 0.0910, 0, 0);
////    distCoeff.push_back(dc);
////    dc =(cv::Mat_<double>(4,1) << -0.1755, 0.1149, 0, 0);
////    distCoeff.push_back(dc);
////
////    cv::Mat tv =(cv::Mat_<double>(3,1) << 0,0,0);
////    transVec.push_back(tv);
////    tv = (cv::Mat_<double>(3,1)<<1327.8,421.1034,463.1997);
////    transVec.push_back(tv);
//
//
//
  /// geometry right-left
    cv::Mat C = (cv::Mat_<double>(3,3)<< 1405.6, 0, 611.8319, 0, 1402.3, 487.3143,0,0,1);
    intrisicMat.push_back(C);
    C = (cv::Mat_<double>(3,3)<< 1398.9,0,639.9772,0,1396.1,471.3643,0,0,1);
    intrisicMat.push_back(C);

    cv::Mat R = (cv::Mat_<double>(3,3)<< 1,0,0,0,1,0,0,0,1);
    rotationMat.push_back(R);
    R = (cv::Mat_<double>(3,3)<< 0.7948, 0.1995, -0.5731, -0.1937, 0.9784, 0.0720, 0.5751, 0.0538, 0.8163);
    rotationMat.push_back(R);

    cv::Mat dc = (cv::Mat_<double>(4,1) << -0.1776, 0.0447, 0, 0);
    distCoeff.push_back(dc);
    dc =(cv::Mat_<double>(4,1) << -0.1625, 0.0571, 0, 0);
    distCoeff.push_back(dc);

    cv::Mat tv =(cv::Mat_<double>(3,1) << 0,0,0);
    transVec.push_back(tv);
    tv = (cv::Mat_<double>(3,1)<<2735.0, -262.0901, 711.3072);
    transVec.push_back(tv);

    ground_plane = (cv::Mat_<double>(3,2) << 0.93674, 0.25936, 0.10902, 0.42199, -0.33263, 0.86871);

//
//
//
//
    sv.SetCamCalibration(intrisicMat, distCoeff, rotationMat, transVec);
    sv.SetGroundPlane(ground_plane);
//
////    sv.HomographyToGround(img_cb, false);
//
    sv.StereoShow(false, filename_traj);
//    sv.DepthShow();

    /// <<<<

    return 0;


}
