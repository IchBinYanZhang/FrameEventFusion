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
    if(argc != 7)
    {
        std::cerr<< "./program [filename1] [filename_timestamp1] [filename2] [filename_timestamp2] [events] [checkerboard image]" <<std::endl;
        return -1;
    }
    std::string filename_frame= argv[1];
    std::string filename_timestamp = argv[2];
    std::string filename_frame2= argv[3];;
    std::string filename_timestamp2 = argv[4];
    std::string filename_events = argv[5];
    std::string filename_cb = argv[6];
    std:: cout << "read frame stream1" <<std::endl;

    FrameStream frames(filename_frame, filename_timestamp);
    std:: cout << "read frame stream2" <<std::endl;

    FrameStream frames2(filename_frame2, filename_timestamp2);

    StereoVision sv (frames, frames2);
    sv.IsSynchronizedTwoStreams();


    /// calibration data
    /// the data is from the Matlab calibration toolbox
    /// the data is specifically for SenseEmotion Experiment 2A (left and middle cam)
    std::vector<cv::Mat> intrisicMat;
    std::vector<cv::Mat> distCoeff;
    std::vector<cv::Mat> rotationMat;
    std::vector<cv::Mat> transVec;

    cv::Mat C = (cv::Mat_<double>(3,3)<< 1405.7, 0, 644.5536, 0, 1402.7, 491.14,0,0,1);
    intrisicMat.push_back(C);
    C = (cv::Mat_<double>(3,3)<< 1403.9,0,640.4511,0,1401.3,475.6392,0,0,1);
    intrisicMat.push_back(C);

    cv::Mat R = (cv::Mat_<double>(3,3)<< 1,0,0,0,1,0,0,0,1);
    rotationMat.push_back(R);
    R = (cv::Mat_<double>(3,3)<< 0.9512,1.1875e-4,-0.3084,-0.0964,0.9500,-0.2969,0.2930,0.3122,0.9037);
    rotationMat.push_back(R);

    cv::Mat dc = (cv::Mat_<double>(4,1) << -0.1557, 0.0910, 0, 0);
    distCoeff.push_back(dc);
    dc =(cv::Mat_<double>(4,1) << -0.1755, 0.1149, 0, 0);
    distCoeff.push_back(dc);

    cv::Mat tv =(cv::Mat_<double>(3,1) << 0,0,0);
    transVec.push_back(tv);
    tv = (cv::Mat_<double>(3,1)<<1327.8,421.1034,463.1997);
    transVec.push_back(tv);


    sv.SetCamCalibration(intrisicMat, distCoeff, rotationMat, transVec);
    cv::Mat img_cb = cv::imread(filename_cb, CV_LOAD_IMAGE_COLOR);

    sv.HomographyToGround(img_cb, false);

    sv.StereoShow(false);
//    sv.DepthShow();

//    EventStream events;
//    events.ReadFromFile(filename_events, 1000000000);
//    FrameEventFuser fuser (frames2, events);
//    fuser.FrameEventShow(false);

    return 0;


}
