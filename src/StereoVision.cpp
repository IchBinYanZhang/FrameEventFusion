#include <opencv2/ximgproc.hpp>
#include "StereoVision.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

StereoVision::StereoVision()
{
    //ctor
}

StereoVision::StereoVision(FrameStream& s1, FrameStream& s2)
{
    std::string filename1;
    std::string filename2;
    std::vector<uint64_t> ts1;
    std::vector<uint64_t> ts2;
//
    s1.GetFrameFile(filename1);
    s1.GetTimestamp(ts1);
    s2.GetFrameFile(filename2);
    s2.GetTimestamp(ts2);
//


    cv::VideoCapture cap1 (filename1);
    cv::VideoCapture cap2 (filename2);

    if(cap1.isOpened() && cap2.isOpened())
        std::cout<< " -- Both frame streams are opened successfully." <<std::endl;
    else{
        std::cerr<< " -- Any frame stream is not opened correctly." <<std::endl;
        exit(-1 );
    }
    this->_stream.push_back(cap1);
    this->_stream.push_back(cap2);


    this->_ts.push_back(ts1);
    this->_ts.push_back(ts2);

    cv::Mat frame;
    cap1 >> frame;

    _nx = frame.cols;
    _ny = frame.rows;

}



void StereoVision::Close(const cv::Mat& src, cv::Mat& dst, int kernelsize)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation
    cv::dilate( src, dst, element );
    cv::erode(dst,dst,element);

}






void StereoVision::SetFrameStream(FrameStream& s1, FrameStream& s2)
{
    std::string filename1;
    std::string filename2;
    std::vector<uint64_t> ts1;
    std::vector<uint64_t> ts2;
//
    s1.GetFrameFile(filename1);
    s1.GetTimestamp(ts1);
    s2.GetFrameFile(filename2);
    s2.GetTimestamp(ts2);
//


    cv::VideoCapture cap1 (filename1);
    cv::VideoCapture cap2 (filename2);

    if(cap1.isOpened() && cap2.isOpened())
        std::cout<< " -- Both frame streams are opened successfully." <<std::endl;
    else{
        std::cerr<< " -- Any frame stream is not opened correctly." <<std::endl;
        exit(-1 );
    }
    this->_stream.push_back(cap1);
    this->_stream.push_back(cap2);


    this->_ts.push_back(ts1);
    this->_ts.push_back(ts2);

}





void StereoVision::SetCamCalibration(std::vector<cv::Mat>& intrisicMat,
                        std::vector<cv::Mat>& distCoeff,
                        std::vector<cv::Mat>& rotationMat,
                        std::vector<cv::Mat>& transVec )

{

    _intrisic_mat = intrisicMat;
    _dist_coeff = distCoeff;
    _rotation_mat = rotationMat;
    _trans_vec = transVec;




    /// computings for image rectification, only for one stereo pair

    cv::Mat R1, P1, R2, P2, Q;


    cv::stereoRectify(_intrisic_mat[0],_dist_coeff[0],_intrisic_mat[1],_dist_coeff[1],
                      cv::Size(_nx,_ny), _rotation_mat[1],_trans_vec[1],
                      R1,R2,P1,P2,Q,CV_CALIB_ZERO_DISPARITY,1, cv::Size(_nx,_ny));


    cv::initUndistortRectifyMap(_intrisic_mat[0], _dist_coeff[0], R1, P1, cv::Size(_nx,_ny),
                                 CV_16SC2, _cam1map1, _cam1map2);
    cv::initUndistortRectifyMap(_intrisic_mat[1], _dist_coeff[1], R2, P2, cv::Size(_nx,_ny),
                                     CV_16SC2, _cam2map1, _cam2map2);



}


void StereoVision::IsSynchronizedTwoStreams()
{

    uint16_t num_frames1 = this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
    uint16_t num_frames2 = this->_stream[1].get(CV_CAP_PROP_FRAME_COUNT);

    std::cout << " - Synchronization checking..." <<std::endl;
    if (num_frames1 != num_frames2)
        std::cout <<" -- The two frame streams are not synchronized. Post-processing \
                         is conducted."<<std::endl;

    if (this->_ts[0].size() != num_frames1 || this->_ts[1].size() != num_frames2 )
        std::cout <<" -- Temporal interpolated frames are used." <<std::endl;

    std::cout << " - Synchronization checking ends." <<std::endl;

}






void StereoVision::ImageRectification(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& out1,
                                      cv::Mat& out2)
{

        cv::remap(frame1, out1, _cam1map1, _cam1map2, cv::INTER_LINEAR);
        cv::remap(frame2, out2, _cam2map1, _cam2map2, cv::INTER_LINEAR);
}


void StereoVision::ImageROIMoG2( cv::Mat& frame, cv::Mat& roi)
/// the ROI is expected to the un-rectified images
/// Method  = Mog2
{
    _pMOG= cv::createBackgroundSubtractorMOG2(50);
    _pMOG->apply(frame, roi);
    this->Close(roi,roi,3);

}


void StereoVision::ImageROIAll( cv::Mat& frame, cv::Mat& roi)
/// the ROI is expected to the un-rectified images
/// Method  = Mog2
{
    int nx = frame.cols;
    int ny = frame.rows;
    roi = 255*cv::Mat::ones(ny,nx,CV_8UC1);

}





float StereoVision::SAD(cv::Mat& set1, cv::Mat& set2)
{
    int nx = set1.cols;
    int ny = set1.rows;
    float dd = 0.0;
    for(int j = 0; j < ny; j++)
    {
        float* ptr1 = set1.ptr<float>(j);
        float* ptr2 = set2.ptr<float>(j);
        for(int i = 0; i < nx; i++)
        {
            dd += fabs(ptr1[i]-ptr2[i]);
        }
    }

    return dd;

}




void StereoVision::BlockMatching(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& frame1_roi, cv::Mat& frame2_roi,cv::Mat& out)
{
    /// Image domain Omega
    int nx = frame1.cols;
    int ny = frame1.rows;
    int i,j;
    frame1.convertTo(frame1,CV_32F);
    frame2.convertTo(frame2,CV_32F);
    frame1_roi.convertTo(frame1_roi, CV_32F);
    frame2_roi.convertTo(frame2_roi, CV_32F);

    out.convertTo(out, CV_32F);




    int d_range = 50;
    int window_radius = 7; // the window size is 2*window_radius+1
    cv::Mat block_f1 = cv::Mat::zeros(2*window_radius+1,2*window_radius+1, CV_32F);
    cv::Mat block_f2 = cv::Mat::zeros(2*window_radius+1,2*window_radius+1, CV_32F);
    std::vector<float> d_stack;
    std::vector<float> error_stack;
    float* ptr_frame1;
    float* ptr_frame2;
    float* ptr_frame1_roi;
    float* ptr_frame2_roi;
    float* ptr_out;





    for(j = window_radius; j < ny-window_radius; j++ )
    {
        ptr_frame1 = frame1.ptr<float>(j);
        ptr_frame2 = frame2.ptr<float>(j);
        ptr_frame1_roi = frame1_roi.ptr<float>(j);
        ptr_frame2_roi = frame2_roi.ptr<float>(j);
        ptr_out = out.ptr<float>(j);

        for( i = window_radius; i < nx-window_radius; i++)
        {
            if (ptr_frame1_roi[i] ==0.0f)
            {
                continue;
            }
            else
            {

                block_f1 = frame1(cv::Rect(i-window_radius, j-window_radius, 2*window_radius+1,2*window_radius+1 ));

                for(int ii = window_radius; ii < nx-window_radius; ii++)
                {
                    if(ptr_frame2_roi[ii] ==0.0f){
                        continue;
                    }

                    else{
                    block_f2 = frame2(cv::Rect(ii-window_radius, j-window_radius, 2*window_radius+1,2*window_radius+1 ));
                    error_stack.push_back(SAD(block_f1, block_f2));
                    d_stack.push_back((float)ii-(float)i   );
                    }
                }

                if(!d_stack.empty())
                {

                    int idx_min = std::distance(error_stack.begin(),std::min_element(error_stack.begin(), error_stack.end()) );
                    ptr_out[i] = d_stack[idx_min];

                    std::cout << "matching pixel (" <<i << ","<<j<<").... disparity = " << ptr_out[i] <<".... SAD=" <<error_stack[idx_min]<<std::endl;
                    d_stack.clear();
                    error_stack.clear();
                }
            }
        }

    }



}




void StereoVision::BlockMatching2(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& frame1_roi, cv::Mat& out)
{
    /// Image domain Omega
    int nx = frame1.cols;
    int ny = frame1.rows;
    cv::Mat segments;
    cv::Mat segments2;

    cv::Ptr<SuperpixelSLIC> su = cv::ximgproc::createSuperpixelSLIC(frame1, SLICO,20, 15);
    cv::Ptr<SuperpixelSLIC> su2 = cv::ximgproc::createSuperpixelSLIC(frame2, SLICO,20, 15);

    su->iterate(10);
    su->enforceLabelConnectivity(25);

    su->getLabelContourMask(segments,false);
    frame1.setTo(Scalar(255),segments);

    su2->iterate(10);
    su2->enforceLabelConnectivity(25);

    su2->getLabelContourMask(segments2,false);
    frame2.setTo(Scalar(255),segments2);


    cv::imshow("pixels", frame1);
    cv::imshow("pixels2", frame2);

        std::cout <<"matching ends"<<std::endl;

    cv::waitKey(10);

}



void StereoVision::DepthEstimate(cv::Mat& frame1, cv::Mat& frame2)
/// frame1 and frame2 are un-rectified images
{

    /// preprocessing, image rectification , ROI
    cv::Mat frame1_roi, frame1_roi_rec,frame1_rec, frame2_rec;
    cv::Mat frame2_roi, frame2_roi_rec;
    this->ImageROIAll(frame1, frame1_roi); // roi is with CV_8UC1
    this->ImageROIAll(frame2, frame2_roi); // roi is with CV_8UC1

    this->ImageRectification(frame1,frame2, frame1_rec, frame2_rec);
    this->ImageRectification(frame1_roi,frame2_roi, frame1_roi_rec, frame2_roi_rec);

    /// disparity map, traditional block matching
    cv::Mat disp_map;
//    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,21);
    cv::cvtColor(frame1_rec, frame1_rec, CV_BGR2GRAY);
    cv::cvtColor(frame2_rec, frame2_rec, CV_BGR2GRAY);

//    bm->compute(frame2_rec, frame1_rec, disp_map);

//    double minval, maxval;
//    cv::minMaxLoc(disp_map,&minval, &maxval);
//    std::cout << minval << "  " <<maxval <<std::endl;


//    disp_map.convertTo(disp_map, CV_8U,255/(maxval-minval),-minval);


    cv::Mat out = cv::Mat::zeros(frame1.size(),CV_8UC1);
    this->BlockMatching(frame1_rec, frame2_rec, frame1_roi_rec, frame2_roi_rec,out);

    cv::normalize(out,out,1,0, NORM_MINMAX);
    frame1_rec.convertTo(frame1_rec, CV_8UC1,1,0);
    frame2_rec.convertTo(frame2_rec, CV_8UC1,1,0);

    cv::imshow("disparity", out);
    cv::imshow("frame2_rec",frame2_rec);
    cv::imshow("frame1_rec", frame1_rec);
    cv::waitKey(0);
//    disp_map.convertTo(disp_map, CV_8U);
//    disp_map = disp_map/16.0f;



}



void StereoVision::DepthShow()
{
    cv::Mat frame1, frame2;
    int num_frames =  this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
    for(int i = 0; i < num_frames; i++)
    {
        _stream[0] >> frame1;
        _stream[1] >> frame2;
        this->DepthEstimate(frame1,frame2);
        std::cout << "matching..."<<std::endl;

    }


}



void StereoVision::StereoShow(bool is_rectified)
{
    std::cout << " - Frame Stream Display." <<std::endl;
    int num_frames =  this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
    cv::namedWindow("stream1", cv::WINDOW_NORMAL);
    cv::namedWindow("stream2", cv::WINDOW_NORMAL);
    cv::Mat frame1, frame1_rec;
    cv::Mat frame2, frame2_rec;

    /// params for rectification
    cv::Mat R1, R2, P1, P2, Q;

    //Create transformation and rectification maps
    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    if( is_rectified)
    {


        cv::stereoRectify(_intrisic_mat[0],_dist_coeff[0],_intrisic_mat[1],_dist_coeff[1],
                          cv::Size(_nx,_ny), _rotation_mat[1],_trans_vec[1],
                          R1,R2,P1,P2,Q,CV_CALIB_ZERO_DISPARITY,1, cv::Size(_nx,_ny));


        cv::initUndistortRectifyMap(_intrisic_mat[0], _dist_coeff[0], R1, P1, cv::Size(_nx,_ny),
                                     CV_16SC2, cam1map1, cam1map2);
        cv::initUndistortRectifyMap(_intrisic_mat[1], _dist_coeff[1], R2, P2, cv::Size(_nx,_ny),
                                     CV_16SC2, cam2map1, cam2map2);



        for(int i = 0; i < num_frames; i++)
        {

            this->_stream[0].read(frame1);
            this->_stream[1].read(frame2);

            if (frame1.cols == 0) {
                std::cout << " -- Error reading frame " << std::endl;
                exit(-1);
            }


            cv::remap(frame1, frame1_rec, cam1map1, cam1map2, cv::INTER_LINEAR);
            cv::remap(frame2, frame2_rec, cam2map1, cam2map2, cv::INTER_LINEAR);



            cv::imshow("stream1",frame1_rec);
            cv::imshow("stream2",frame2_rec);
            cv::waitKey(30);
        }

    }
    else
    {
        for(int i = 0; i < num_frames; i++)
        {

            this->_stream[0].read( frame1);
            this->_stream[1].read(frame2);

            cv::imshow("stream1",frame1);
            cv::imshow("stream2",frame2);

            cv::waitKey(50);
        }

    }


    std::cout << " - Frame Stream Display Ends." <<std::endl;

}






StereoVision::~StereoVision()
{
    //dtor
}
