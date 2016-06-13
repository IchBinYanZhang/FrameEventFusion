#include <opencv2/ximgproc.hpp>
#include "StereoVision.h"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

using namespace std;
using namespace cv;

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


void StereoVision::Open(const cv::Mat& src, cv::Mat& dst, int kernelsize)
// kernelsize is only odd!
{

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                   cv::Size( kernelsize, kernelsize ),
                                   cv::Point( (kernelsize-1)/2, (kernelsize-1)/2 ) );
    // Apply the dilation operation

    cv::erode(src,dst,element);
    cv::dilate( dst, dst, element );
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


void StereoVision::HomographyToGround(cv::Mat& img_cb, bool img_show)
{
    int nx = 5;
    int ny = 7;
    cv::Size pattern_size(nx,ny);
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> corners_w;
    bool patternfound = cv::findChessboardCorners(img_cb, pattern_size, corners,
        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);

    for(auto i = corners.begin(); i < corners.end(); i++)
    {
        double x;
        double y;
        int N = std::distance(corners.begin(), i);
        x = N%nx;
        y = (N-x)/nx;
        corners_w.push_back(cv::Point2f(5*x+img_cb.cols/4,5*y+3*img_cb.rows/4));

    }

    _H = cv::findHomography(corners,corners_w);


    if(img_show)
    {

        cv::drawChessboardCorners(img_cb, pattern_size, Mat(corners), patternfound);
        cv::namedWindow("corners",WINDOW_NORMAL);
        imshow("corners", img_cb);
        cv::waitKey(0);
    }
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


inline void StereoVision::MultiBoundingBoxFromROI(cv::Mat& roi, std::vector<cv::Rect>& boundRect)
/// multiple object bounding box
{

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::RNG rng(12345);
    cv::Mat roi_w;
    roi.convertTo(roi_w,CV_8UC1);

    /// Find contours
    findContours( roi_w, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );

    for( int i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }



}






inline void StereoVision::SingleBoundingBoxFromROI(cv::Mat& roi, cv::Rect& bd)
/// single object bounding box
{

    std::vector<cv::Point> pts;
    cv::Mat roi_w;
    roi.convertTo(roi_w,CV_8UC1);

    /// Find points
    for(int i = 0; i < roi_w.rows; i++)
    {
        uint8_t* ptr = roi_w.ptr<uint8_t>(i);
        for(int j = 0; j < roi_w.cols; j++)
        {
            if(ptr[j] !=0)
                pts.push_back(cv::Point(j,i));
        }
    }

    /// Approximate contours to polygons + get bounding rects and circles
    bd = boundingRect( pts );

}



inline void StereoVision::ShowBoundingBox(const cv::Mat& drawing, cv::Rect& bd)
{
        cv::RNG rng(12345);
        cv::Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::rectangle( drawing, bd.tl(), bd.br(), color, 2, 8, 0 );
        cv::Point2f ptt = (bd.br()+bd.tl())/2;

        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_NORMAL );
        imshow( "Contours", drawing );
}



inline void StereoVision::ShowTracking(const cv::Mat& f_current, cv::Rect& bd, std::vector<cv::Point2f>& trajectory, cv::Mat& out)
{

        f_current.copyTo(out);
//        cv::RNG rng(12345);
        cv::Scalar color = Scalar( 36, 150, 223 );
        cv::rectangle( out, bd.tl(), bd.br(), color, 2, 8, 0 );
        for(int i = 0; i < trajectory.size()-1; i++){
            cv::line(out, trajectory[i], trajectory[i+1], color,2, CV_AA);
        }


}



inline void StereoVision::ShowBoundingBox(const cv::Mat& drawing, std::vector<cv::Rect>& bd)
{
        cv::RNG rng(12345);
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

        for( int i = 0; i< bd.size(); i++ )
        {
           Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           rectangle( drawing, bd[i].tl(), bd[i].br(), color, 2, 8, 0 );
        }

        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );
}










inline void StereoVision::FundamentalMatrixFromCalibration(const cv::Mat& K, const cv::Mat& R, const cv::Mat& T,
                                                           const cv::Mat& K_prime, const cv::Mat& R_prime, const cv::Mat& T_prime,
                                                           cv::Mat& F)
/// the fundamental matrix is to measure <x',Fx>=0, assuming that the direction is from x to x'
{

    /// move the origin so that P = K[I|0] and P' = K[_R|_T]
    cv::Mat _R, _T, _T_cross; // relative transform from C to C', so that P = K[I|0] and P' = K[_R|_T]
    _R = R_prime * R.inv();
    _T = T_prime - T;
    _T_cross = (cv::Mat_<double>(3,3)<< 0, -_T.at<double>(2), _T.at<double>(1), _T.at<double>(2), 0, -_T.at<double>(0), -_T.at<double>(1),_T.at<double>(0),0);

    F = (K_prime.inv()).t() * _T_cross * _R * K.inv();

}










inline void StereoVision::HistDisplay(const cv::Mat& hist, const int nbins, const float* histRange )
{
    int hist_h = 512;
    int hist_w = 400;
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat hist_out;
    int bin_w = cvRound( (double) hist_w/nbins );
  /// Normalize the result to [ 0, histImage.rows ]
  normalize(hist, hist_out, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


  /// Draw for each channel
  for( int i = 1; i < nbins; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist_out.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist_out.at<float>(i)) ),
                       Scalar( 255, 255, 0), 2, 8, 0  );
  }
  imshow("histogram", histImage);
}



inline void StereoVision::MaskFromRect(const cv::Mat& img, cv::Rect& bd, cv::Mat& out)
{
    out = cv::Mat::zeros(img.size(), img.depth());
    vector<Point> bd_vertices {Point(bd.x, bd.y), Point(bd.x+bd.width, bd.y),Point(bd.x+bd.width, bd.y+bd.height),Point(bd.x, bd.y+bd.height)};
    vector<Point> bd_poly;
    approxPolyDP(bd_vertices, bd_poly, 1.0, true);
    fillConvexPoly(out, &bd_poly[0], (int)bd_poly.size(), 255, 8, 0);
}





//
//bool StereoVision::Tracking2D( const cv::Mat& f0, const cv::Mat& f1, cv::Rect& bd, cv::Mat& hist, Tracking2DMethod method)
///// f1, f0 the current frame and the previous frame
///// bd, in parameters
///// hist, in
///// bd_new, out the new bounding box
//{
//
//    cv::Mat roi;
//    bool found_box = false;
//
//
//    Tracking2DMethod _method = TRACKINGBYDETECTION;
//    Mat image, image_h, image_s, image_v,image_lbp, mask = Mat::zeros(f1.size(), CV_8UC1), backproj;
//
//    /// histogram parameters lbp
//    int imgCount = 3;
//    int dims = 3;
//    const int sizes[] = {90,128,128};
//    const int channels[] = {0,1,2};
//    float hRange[] = {0,180};
//    float vRange [] = {0,256};
//    float lbpRange[] = {0,256};
//    const float *ranges[] = { hRange,vRange, lbpRange};
//    float radius = 5;
//    int n_points = 8;
//
//
//    /// histogram parameters of HSV
//
////    int imgCount = 1;
////    int dims = 2;
////    const int sizes[] = {30,30};
////    const int channels[] = {0,1};
////    float hRange[] = {0,180};
////    float sRange[] = {0,256};
////    const float *ranges[] = {hRange, sRange};
////    float radius = 1;
////    int n_points = 8;
//
//    switch(method)
//    {
//
//
//        case TRACKINGBYDETECTION:
//        /// it directly gives us the bounding box. This can be used for tracking initialization.
//        {
//
//            StipDetector dec (f0, f1);
//            dec.DefineROI();
//            dec.GetROI(roi);
////            cv::imshow("roi",roi);
//            roi.convertTo(roi,CV_8UC1);
//            double minval, maxval;
//            cv::minMaxLoc(roi,&minval, &maxval);
//
//            if(minval == maxval)
//            {
//                std::cout << minval << " " <<maxval <<std::endl;
//                std::cout << "no motion is detected" <<std::endl;
//            }
//            else
//            {
//                /// compute the bounding box
//                SingleBoundingBoxFromROI(roi, bd);
//                found_box = true;
//
//                /// compute the histogram
//
//                //CLBP
//
//
////                cvtColor(f1, image, COLOR_BGR2GRAY);
////                CLBP(image, image_lbp, radius, n_points);
////
////                //HSV
////                cvtColor(f1,image,COLOR_BGR2HSV);
////                int ch1[] = {0, 0};
////                int ch2[] = {2, 0};
////
////                image_s.create(image.size(), image.depth());
////                mixChannels(&image, 1, &image_s, 1,ch1, 1);
////                image_v.create(image.size(), image.depth());
////                mixChannels(&image, 1, &image_v, 1, ch2, 1);
////
////
////
////                cv::Mat imgSet [] = {image_s,image_v, image_lbp};
//
//                vector<Point> bd_vertices {Point(bd.x, bd.y), Point(bd.x+bd.width, bd.y),Point(bd.x+bd.width, bd.y+bd.height),Point(bd.x, bd.y+bd.height)};
//                vector<Point> bd_poly;
//                approxPolyDP(bd_vertices, bd_poly, 1.0, true);
//                fillConvexPoly(mask, &bd_poly[0], (int)bd_poly.size(), 255, 8, 0);
//
//
////                Mat image_lbp_roi (image_lbp, bd);
////                Mat mask_roi(mask, bd);
//                calcHist(imgSet, imgCount, channels, mask, hist, dims, sizes, ranges);
////                normalize(hist, hist, 0, 255, NORM_MINMAX);
//            }
//
//            break;
//        }
//        case CAMSHIFT:
//        /// update the bounding box using the camshift algorithm, in which the size,location and rotation will update at the same time.
//        {
//
//            /// extract motion
//            cv::Rect bd_motion;
//            StipDetector dec (f0, f1);
//            dec.DefineROI();
//            dec.GetROI(roi);
//            roi.convertTo(roi,CV_8UC1);
//            SingleBoundingBoxFromROI(roi, bd_motion);
//
//            if (bd_motion.width!=0 && bd_motion.height!=0)
//            {
//
//
//                ///extract histogram
//                cvtColor(f1, image, COLOR_BGR2GRAY);
//                CLBP(image,image_lbp, radius, n_points);
//
//                cvtColor(f1, image, COLOR_BGR2HSV);
//                int ch1[] = {0, 0};
//                int ch2[] = {2, 0};
//                image_s.create(image.size(), image.depth());
//                mixChannels(&image, 1, &image_s, 1,ch1, 1);
//                image_v.create(image.size(), image.depth());
//                mixChannels(&image, 1, &image_v, 1, ch2, 1);
//
//
//                cv::Mat imgSet[] = {image_s,image_v, image_lbp};
//
//                cv::calcBackProject(imgSet,imgCount,channels,hist,backproj,ranges);
//                cv::GaussianBlur(backproj, backproj, cv::Size(0,0), 3.5,3.5, BORDER_REFLECT);
//
//                cv::Mat mask_motion;
//                MaskFromRect(f1, bd_motion, mask_motion);
//                backproj &= mask_motion;
//
//                RotatedRect trackBox = CamShift(backproj, bd,
//                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
//                normalize(backproj,backproj,0,255,NORM_MINMAX);
//                imshow("backproj", backproj);
//                bd = trackBox.boundingRect();
//                MaskFromRect(f1, bd, mask_motion);
//
//                calcHist(imgSet, imgCount, channels, mask_motion, hist, dims, sizes, ranges);
//
//            }
//            else
//            {
//
//            }
//
//
//            found_box = true;
//            break;
//        }
//
//
//        default:
//            break;
//    }
//
////    HistDisplay(hist, sizes[0], ranges[0] );
//    return found_box;
//
//}


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





inline float StereoVision::SAD(cv::Mat& set1, cv::Mat& set2)
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
    int window_radius = 10; // the window size is 2*window_radius+1
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



void StereoVision::DenseDepthEstimate(cv::Mat& frame1, cv::Mat& frame2)
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
        this->DenseDepthEstimate(frame1,frame2);
        std::cout << "matching..."<<std::endl;

    }


}


void StereoVision::PointSetPerspectiveTransform(const std::vector<Point2f>& in, std::vector<Point2f>& out, cv::Mat& H )
/// this function aims to project one point set to another one, based on the homography H
/// The points are 2D, written as (x,y), while the homography is based on the homogeneous coordinate.
{
    out.clear();
    int N = in.size();
    cv::Matx33f _H (H);
    for(int i = 0; i < N; i++)
    {
        Point3f _pt = Point3f(in[i].x, in[i].y, 1.0);
        Point3f _pth = _H*_pt;
        out.push_back(Point2f(_pth.x/_pth.z, _pth.y/_pth.z));
//        cout << in[i] <<endl;
    }
}


inline void StereoVision::UpdateIntrinsicByImageResize(cv::Mat& src, cv::Mat& dst, double fx, double fy)
/// image is resized by factor fx in x-direction and fy in y-direction
{
    cv::Mat factor_mat = (Mat_<double>(3,3) << fx, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 1.0);
    dst = factor_mat * src;
}


void StereoVision::StereoShow(bool is_rectified)
{
    std::cout << " - Frame Stream Display." <<std::endl;
    int num_frames =  this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
//    cv::namedWindow("stream1", cv::WINDOW_NORMAL);
//    cv::namedWindow("stream2", cv::WINDOW_NORMAL);
    cv::Mat frame1, frame1_rec, frame1_pre, frame1_birdview, frame1_s, frame1_tracking,frame1_birdview_tracking;
    cv::Mat frame2, frame2_rec, frame2_pre, frame2_birdview, frame2_s, frame2_tracking, frame2_birdview_tracking;
    std::vector<cv::Point2f> trajectory1;
    std::vector<cv::Point2f> trajectory1_bird;
    std::vector<cv::Point2f> trajectory2;
    std::vector<cv::Point2f> trajectory2_bird;



    /// params for rectification
    cv::Mat R1, R2, P1, P2, Q;

    //Create transformation and rectification maps
    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    /// for tracking
    cv::Rect bd1, bd2;
    cv::Mat hist1, hist2;

    double resize_factor_x = 0.5, resize_factor_y = 0.5;
    UpdateIntrinsicByImageResize(_intrisic_mat[0], _intrisic_mat[0], resize_factor_x, resize_factor_y);
    UpdateIntrinsicByImageResize(_intrisic_mat[1], _intrisic_mat[1], resize_factor_x, resize_factor_y);

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

            resize(frame2,frame2,Size(0,0), resize_factor_x, resize_factor_y);
            resize(frame1,frame1,Size(0,0), resize_factor_x, resize_factor_y);

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
        trajectory1.clear();
        trajectory1_bird.clear();
        trajectory2.clear();
        trajectory2_bird.clear();


        for(int i = 0; i < num_frames; i++)
        {

            this->_stream[0].read(frame1);
            this->_stream[1].read(frame2);

            if(frame2.empty())
            {
                cout<< "----------Display Ends--------------"<<endl;
            }
            else{
                cout << "------------frame "<<i<<" -----------------"<<endl;
                resize(frame2,frame2,Size(0,0), resize_factor_x, resize_factor_y);
                resize(frame1,frame1,Size(0,0), resize_factor_x, resize_factor_y);

//                cv::imshow("stream1",frame1);
//                cv::imshow("stream2",frame2);
//                cv::namedWindow("birdview", CV_WINDOW_NORMAL);
//                ImagePreprocessing(frame1, frame1_s); // only gaussian blur
//                ImagePreprocessing(frame2, frame2_s); // only gaussian blur

//                if(i>120  )
//                {
//                    if(bd1.height == 0.0 || bd1.width == 0.0 || bd2.height == 0.0 || bd2.width == 0.0){
//                        cout << " -- detection" <<endl;
//                        Tracking2D(frame1_pre, frame1_s, bd1, hist1, TRACKINGBYDETECTION);
//                        Tracking2D(frame2_pre, frame2_s, bd2, hist2, TRACKINGBYDETECTION);
//
//                    }
//                    else
//                    {
//                        cout << " -- tracking" <<endl;
//
//                        Tracking2D(frame1_pre, frame1_s, bd1, hist1,CAMSHIFT);
//                        trajectory1.push_back( Point2f((bd1.tl()+bd1.br())/2) );
//                        ShowTracking(frame1,bd1, trajectory1, frame1_tracking);
//    //                    cv::warpPerspective(frame2_visual, frame2_birdview, _H, cv::Size(frame1.cols, frame1.rows));
//
//                        Tracking2D(frame2_pre, frame2_s, bd2, hist2,CAMSHIFT);
//                        trajectory2.push_back( Point2f((bd2.tl()+bd2.br())/2) );
//                        ShowTracking(frame2,bd2, trajectory2, frame2_tracking);
//                        // int the bird view
////                        cv::warpPerspective(frame2, frame2_birdview, _H, cv::Size(frame1.cols, frame1.rows));
////                        PointSetPerspectiveTransform(trajectory, trajectory_bird, _H );
////                        ShowTracking(frame2_birdview, bd_new, trajectory_bird, frame2_birdview_tracking);
//                        cv::imshow("stream1_tracking", frame1_tracking);
//                        cv::imshow("stream2_tracking", frame2_tracking);
//
//    //                    cv::waitKey(0);
//                    }
//                }
//                frame1_s = frame1;
//                frame2_s = frame2;
                ImagePreprocessing(frame1, frame1_s); // only gaussian blur
                ImagePreprocessing(frame2, frame2_s); // only gaussian blur

                if(i>50)
                {


                    if(bd1.height == 0.0 || bd1.width == 0.0 || bd2.height == 0.0 || bd2.width == 0.0)
                    {
                        cout << " -- detection" <<endl;
                        Tracking3D(frame1_pre, frame1_s, bd1, hist1, frame2_pre, frame2_s, bd2, hist2, TRACKINGBYDETECTION3, HSVLBP);
                    }



                    else
                    {
                        cout << " -- tracking" <<endl;
                        namedWindow("stream1_tracking", CV_WINDOW_NORMAL);
                        namedWindow("stream2_tracking", CV_WINDOW_NORMAL);
                        moveWindow("stream1_tracking", 0,0);
                        moveWindow("stream2_tracking", 0,700);



                        Tracking3D(frame1_pre, frame1_s, bd1, hist1, frame2_pre, frame2_s, bd2, hist2, EPICAMSHIFT, HSVLBP);
                        trajectory1.push_back( Point2f((bd1.tl()+bd1.br())/2) );
                        ShowTracking(frame1,bd1, trajectory1, frame1_tracking);

                        trajectory2.push_back( Point2f((bd2.tl()+bd2.br())/2) );
                        ShowTracking(frame2,bd2, trajectory2, frame2_tracking);

                        cv::imshow("stream1_tracking", frame1_tracking);
                        cv::imshow("stream2_tracking", frame2_tracking);
                        cv::waitKey(5);


                    }

                }


                frame1_pre = frame1_s.clone();
                frame2_pre = frame2_s.clone();

            }
        }

    }



}


inline void GetHog(const cv::Mat& in, cv::Rect& bd, cv::Mat&hist)
{

    cv::Mat img = in(bd);
    cvtColor(img,img,COLOR_BGR2GRAY);
    HOGDescriptor hog;
    vector<float> ders;
    vector<Point>locs;
    hog.compute(img,ders,Size(32,32),Size(0,0),locs);

    hist.create(ders.size(),1,CV_32FC1);

    for(int i=0;i<ders.size();i++)
    {
      hist.at<float>(i,0)=ders.at(i);

    }

}



bool StereoVision::Tracking3D(const cv::Mat& f0_pre, const cv::Mat& f0_cur, cv::Rect& bd0, cv::Mat& hist0,
                              const cv::Mat& f1_pre, const cv::Mat& f1_cur, cv::Rect& bd1, cv::Mat& hist1,
                              Tracking3DMethod method, FeatureMethod method_feature)
{

    cv::Mat roi0, roi1; // region of interest
    bool found_box = false;


    Tracking3DMethod _method = TRACKINGBYDETECTION3;
    Mat image0, image0_h, image0_s, image0_v,image0_lbp, image0_lbp1,image0_lbp2,mask = Mat::zeros(f0_cur.size(), CV_8UC1);
    Mat image1, image1_h, image1_s, image1_v,image1_lbp,image1_lbp1,image1_lbp2;




    ///Local binary pattern setting

    int n_scales = 3;
    int radius = 1;
    int n_points = 8;
    LocalBinaryPattern pattern0 (radius, n_points, false);
    LocalBinaryPattern pattern1 (2*radius, 2*n_points, false);
    LocalBinaryPattern pattern2 (3*radius, 3*n_points, false);



    /// histogram parameters
    int imgCount = 1;
    int dims = 1;
    const int bin0 = pattern0.GetNumBins();
    const int bin1 = pattern1.GetNumBins();
    const int bin2 = pattern2.GetNumBins();


    int range0 = pattern0.GetRanges();
    int range1 = pattern1.GetRanges();
    int range2 = pattern2.GetRanges();

    cout << range0<<endl;
    const int sizes[] = {bin0/3};
    const int channels[] = {0,1};
    float hRange[] = {0,128};
    float sRange [] = {0,256};
    float vRange [] = {0,256};
    float lbp0Range[] = {0,range0};
    float lbp1Range[] = {0,range1};
    float lbp2Range[] = {0,range2};


    const float *ranges[] = { lbp0Range};


    int ch1[] = {0, 0};
    int ch2[] = {1, 0};
    int ch3[] = {2, 0};







    switch(method)
    {

         case TRACKINGBYDETECTION3:
        /// it directly gives us the bounding box. This can be used for tracking initialization.
        {
            cv::Rect bd0_tmp, bd1_tmp;

            StipDetector dec0 (f0_pre, f0_cur);
            dec0.DefineROI();
            dec0.GetROI(roi0);
            roi0.convertTo(roi0,CV_8UC1);
            SingleBoundingBoxFromROI(roi0, bd0_tmp);

            StipDetector dec1 (f1_pre, f1_cur);
            dec1.DefineROI();
            dec1.GetROI(roi1);
            roi1.convertTo(roi1,CV_8UC1);
            SingleBoundingBoxFromROI(roi1, bd1_tmp);


            if(bd0_tmp.area() <= 8000 || bd1_tmp.area() <= 8000)
            {
                std::cout << " -- no motion is detected." <<std::endl;
            }
            else
            {
                /// compute the bounding box
                bd0 = bd0_tmp;
                bd1 = bd1_tmp;

                found_box = true;



                if(method_feature==HSVLBP){



                /// compute the histogram

                //CLBP
                cvtColor(f0_cur, image0, COLOR_BGR2GRAY);
                pattern0.UniformLBP(image0, image0_lbp);
                pattern1.UniformLBP(image0, image0_lbp1);
                pattern2.UniformLBP(image0, image0_lbp2);

                cvtColor(f1_cur, image1, COLOR_BGR2GRAY);
                pattern0.UniformLBP(image1, image1_lbp);
                pattern1.UniformLBP(image1, image1_lbp1);
                pattern2.UniformLBP(image1, image1_lbp2);


                //HSV
//                cvtColor(f0_cur,image0,COLOR_BGR2HSV);
//                cvtColor(f1_cur,image1,COLOR_BGR2HSV);
//                //RGB
////                image0 = f0_cur.clone();
////                image1 = f1_cur.clone();
//
//                image0_h.create(image0.size(), image0.depth());
//                mixChannels(&image0, 1, &image0_h, 1,ch1, 1);
//                image0_s.create(image0.size(), image0.depth());
//                mixChannels(&image0, 1, &image0_s, 1, ch2, 1);
//                image0_v.create(image0.size(), image0.depth());
//                mixChannels(&image0, 1, &image0_v, 1, ch3, 1);
//
//                image1_h.create(image1.size(), image1.depth());
//                mixChannels(&image1, 1, &image1_h, 1,ch1, 1);
//                image1_s.create(image1.size(), image1.depth());
//                mixChannels(&image1, 1, &image1_s, 1, ch2, 1);
//                image1_v.create(image1.size(), image1.depth());
//                mixChannels(&image1, 1, &image1_v, 1, ch3, 1);



//                image1_s.create(image1.size(), image1.depth());
//                mixChannels(&image1, 1, &image1_s, 1,ch1, 1);
//                image1_v.create(image1.size(), image1.depth());
//                mixChannels(&image1, 1, &image1_v, 1, ch2, 1);

//                imshow("lbp0", image0_lbp);
//                imshow("lbp1", image0_lbp1);
//                imshow("lbp2", image0_lbp2);
//                waitKey(0);



//                normalize(image0_lbp, image0_lbp, 0,255,NORM_MINMAX);
//                normalize(image0_lbp1, image0_lbp1, 0,255,NORM_MINMAX);
//
//                normalize(image1_lbp, image1_lbp, 0,255,NORM_MINMAX);
//                normalize(image1_lbp1, image1_lbp1, 0,255,NORM_MINMAX);

//                image0_lbp.convertTo(image0_lbp, CV_8UC1);
//                image1_lbp.convertTo(image1_lbp, CV_8UC1);
                cv::Mat imgSet0 [] = {image0_lbp};
                cv::Mat imgSet1 [] = {image1_lbp};





                vector<Point> bd_vertices0 {Point(bd0.x, bd0.y), Point(bd0.x+bd0.width, bd0.y),Point(bd0.x+bd0.width, bd0.y+bd0.height),Point(bd0.x, bd0.y+bd0.height)};
                vector<Point> bd_poly0;
                approxPolyDP(bd_vertices0, bd_poly0, 1.0, true);

                vector<Point> bd_vertices1 {Point(bd1.x, bd1.y), Point(bd1.x+bd1.width, bd1.y),Point(bd1.x+bd1.width, bd1.y+bd1.height),Point(bd1.x, bd1.y+bd1.height)};
                vector<Point> bd_poly1;
                approxPolyDP(bd_vertices1, bd_poly1, 1.0, true);


                fillConvexPoly(mask, &bd_poly0[0], (int)bd_poly0.size(), 255, 8, 0);
                calcHist(imgSet0, imgCount, channels, mask, hist0, dims, sizes, ranges);

                imshow("mask", mask);
                cout << mask.depth()<<endl;
                waitKey(0);


                fillConvexPoly(mask, &bd_poly1[0], (int)bd_poly1.size(), 255, 8, 0);
                calcHist(imgSet1, imgCount, channels, mask, hist1, dims, sizes, ranges);

                normalize(hist0, hist0, 0,1,NORM_MINMAX);
                normalize(hist1, hist1, 0,1,NORM_MINMAX);

                }
                else{
                    if(method_feature==HOG){
                        GetHog(f0_cur,bd0,hist0);
                        GetHog(f1_cur,bd1,hist1);

                    }
                }

            }

            break;
        }

        case EPICAMSHIFT:
        {

            namedWindow("color0", CV_WINDOW_NORMAL);
            namedWindow("color1", CV_WINDOW_NORMAL);
            namedWindow("epipolar0", CV_WINDOW_NORMAL);
            namedWindow("epipolar1", CV_WINDOW_NORMAL);
            namedWindow("afterwards0",CV_WINDOW_NORMAL);
            namedWindow("afterwards1",CV_WINDOW_NORMAL);

            int posx=1000, posy=0;
            cv::moveWindow("color0", posx+0, posy+0);
            cv::moveWindow("color1", posx+700, posy+0);
            cv::moveWindow("epipolar0", posx+0, posy+700);
            cv::moveWindow("epipolar1", posx+700, posy+700);
            cv::moveWindow("afterwards0", posx+0, posy+1400);
            cv::moveWindow("afterwards1", posx+700, posy+1400);








            /// extract motion
            cout << "-- extract motion roi" <<endl;
            cv::Rect bd0_motion, bd1_motion;
            cv::Mat mask0_motion, mask1_motion;

            StipDetector dec0 (f0_pre, f0_cur);
            dec0.DefineROI();
            dec0.GetROI(roi0);
            roi0.convertTo(roi0,CV_8UC1);


            StipDetector dec1 (f1_pre, f1_cur);
            dec1.DefineROI();
            dec1.GetROI(roi1);
            roi1.convertTo(roi1,CV_8UC1);
            SingleBoundingBoxFromROI(roi0, bd0_motion);
            SingleBoundingBoxFromROI(roi1, bd1_motion);

            if(bd0_motion.area()<bd0.area() || bd1_motion.area()<bd1.area()){
                bd0_motion = bd0;
                bd1_motion = bd1;
            }


            /// fundemental matrix
            cv::Mat pdf_epi0, pdf_epi1;
            cv::Mat F01, F10; // from image 0 to image1 || from image 1 to image 0
            cv::Mat backproj0, backproj1;
            cv::Mat post0,post1;
            FundamentalMatrixFromCalibration(_intrisic_mat[0], _rotation_mat[0], _trans_vec[0],
                                             _intrisic_mat[1], _rotation_mat[1], _trans_vec[1],
                                             F01);

            FundamentalMatrixFromCalibration(_intrisic_mat[1], _rotation_mat[1], _trans_vec[1],
                                             _intrisic_mat[0], _rotation_mat[0], _trans_vec[0],
                                             F10);



            F01.convertTo(F01,CV_32F);
            F10.convertTo(F10,CV_32F);

            cout << "-- extract epipolar constraint" <<endl;




            if (bd0_motion.area()!=0 && bd1_motion.area()!=0)
            {




            ///extract histogram and back projection pdf

            //CLBP
                cvtColor(f0_cur, image0, COLOR_BGR2GRAY);
                pattern0.UniformLBP(image0, image0_lbp);
                pattern1.UniformLBP(image0, image0_lbp1);
                pattern2.UniformLBP(image0, image0_lbp2);

                cvtColor(f1_cur, image1, COLOR_BGR2GRAY);
                pattern0.UniformLBP(image1, image1_lbp);
                pattern1.UniformLBP(image1, image1_lbp1);
                pattern2.UniformLBP(image1, image1_lbp2);

//                normalize(image0_lbp, image0_lbp, 0,255,NORM_MINMAX);
//                normalize(image0_lbp1, image0_lbp1, 0,255,NORM_MINMAX);
//
//                normalize(image1_lbp, image1_lbp, 0,255,NORM_MINMAX);
//                normalize(image1_lbp1, image1_lbp1, 0,255,NORM_MINMAX);



//                image0_lbp.convertTo(image0_lbp, CV_8UC1);
//                image1_lbp.convertTo(image1_lbp, CV_8UC1);
//                image0.convertTo(image0,CV_8UC1);
                cv::Mat imgSet0 [] = {image0_lbp};
                cv::Mat imgSet1 [] = {image1_lbp};

                imshow("image0", image0);
                imshow("lbp0", image0_lbp);
                imshow("lbp1", image0_lbp1);

//            cvtColor(f0_cur, image0, COLOR_BGR2GRAY);
//            CLBP(image0,image0_lbp, radius, n_points);
//
//            cvtColor(f1_cur, image1, COLOR_BGR2GRAY);
//            CLBP(image1,image1_lbp, radius, n_points);
//
//
//            ///hsv
//            cvtColor(f0_cur, image0, COLOR_BGR2HSV);
//            cvtColor(f1_cur, image1, COLOR_BGR2HSV);
//
//
//            ///rgb
////            image0 = f0_cur.clone();
////            image1 = f1_cur.clone();
//
//            image0_h.create(image0.size(), image0.depth());
//            mixChannels(&image0, 1, &image0_h, 1,ch1, 1);
//            image0_s.create(image0.size(), image0.depth());
//            mixChannels(&image0, 1, &image0_s, 1, ch2, 1);
//            image0_v.create(image0.size(), image0.depth());
//            mixChannels(&image0, 1, &image0_v, 1, ch3, 1);
//
//            image1_h.create(image1.size(), image1.depth());
//            mixChannels(&image1, 1, &image1_h, 1,ch1, 1);
//            image1_s.create(image1.size(), image1.depth());
//            mixChannels(&image1, 1, &image1_s, 1, ch2, 1);
//            image1_v.create(image1.size(), image1.depth());
//            mixChannels(&image1, 1, &image1_v, 1, ch3, 1);
//
//
//
////                image1_s.create(image1.size(), image1.depth());
////                mixChannels(&image1, 1, &image1_s, 1,ch1, 1);
////                image1_v.create(image1.size(), image1.depth());
////                mixChannels(&image1, 1, &image1_v, 1, ch2, 1);
//
//
//
//            cv::Mat imgSet0 [] = {image0_h, image0_s,image0_v, image0_lbp};
//            cv::Mat imgSet1 [] = {image0_h, image1_s,image1_v, image1_lbp};


            cv::calcBackProject(imgSet0,imgCount,channels,hist0,backproj0,ranges);
            cv::calcBackProject(imgSet1,imgCount,channels,hist1,backproj1,ranges);


            cout << backproj0.depth() << " " << hist0.depth() <<endl;

            backproj0.convertTo(backproj0, CV_32F); // normalize backproj as a pdf
            backproj1.convertTo(backproj1, CV_32F);
            normalize(backproj0, backproj0,0,1,NORM_MINMAX);
            normalize(backproj1, backproj1,0,1,NORM_MINMAX);

            cv::GaussianBlur(backproj0, backproj0, cv::Size(0,0), 2.5,2.5, BORDER_REFLECT);
            cv::GaussianBlur(backproj1, backproj1, cv::Size(0,0), 2.5,2.5, BORDER_REFLECT);

            cv::imshow("color0", backproj0);
            cv::imshow("color1", backproj1);
            cout << "-- extract motion roi" <<endl;



//            imshow("lbp", image0_h);
//            imshow("s", image0_s);
//            imshow("v", image0_v);
//            imshow("lbp", image0_lbp);



            /// pdf from epipolar geometry
            pdf_epi1 = Mat::zeros(backproj1.size(), CV_32F );
            pdf_epi0 = Mat::zeros(backproj0.size(), CV_32F );
            for(int k = 0; k < 3; k++)
            {


                cout << "--- update epipolar constraint" <<endl;

                cv::Point2f cen_bd0( (bd0.tl()+bd0.br())/2.0   );
                cv::Point2f cen_bd1( (bd1.tl()+bd1.br())/2.0   );

                cv::Mat cen_bd0_homo = (Mat_<float>(3,1)<< float(cen_bd0.x), float(cen_bd0.y), 1.0f);
                cv::Mat cen_bd1_homo = (Mat_<float>(3,1)<< float(cen_bd1.x), float(cen_bd1.y), 1.0f);




                cv::Mat epipolar_line1 = F10*cen_bd1_homo;
                cv::Mat epipolar_line0 = F01*cen_bd0_homo;



                for(int i = 0; i < backproj1.rows; i++)
                    for(int j = 0; j < backproj1.cols; j++)
                {
                    pdf_epi0.at<float>(i,j) = (fabs(epipolar_line1.at<float>(0,0)*float(j) + epipolar_line1.at<float>(1,0)*float(i) + epipolar_line1.at<float>(2,0))<1.0)?1.0f : 0.0f;
                    pdf_epi1.at<float>(i,j) = (fabs(epipolar_line0.at<float>(0,0)*float(j) + epipolar_line0.at<float>(1,0)*float(i) + epipolar_line0.at<float>(2,0))<1.0)?1.0f : 0.0f;

                }

                cv::GaussianBlur(pdf_epi0, pdf_epi0, cv::Size(0,0), 0.01,18.5, BORDER_REFLECT);
                cv::GaussianBlur(pdf_epi1, pdf_epi1, cv::Size(0,0), 0.01,18.5, BORDER_REFLECT);



                imshow("epipolar0", pdf_epi0);
                imshow("epipolar1", pdf_epi1);
//
//                waitKey(5);
                post0 = backproj0.mul(pdf_epi0);
                post1 = backproj1.mul(pdf_epi1);


                post0.convertTo(post0,CV_8UC1, 255);
                post1.convertTo(post1,CV_8UC1, 255);


                MaskFromRect(f0_cur, bd0_motion, mask0_motion);
                MaskFromRect(f1_cur, bd1_motion, mask1_motion);
                post0 &= mask0_motion;
                post1 &= mask1_motion;


                /// meanshift update
                meanShift(post0,bd0, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                meanShift(post1,bd1, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));


            }
//            RotatedRect trackBox0 = CamShift(post0, bd0,
//                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
//
//            RotatedRect trackBox1 = CamShift(post1, bd1,
//                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
//
//
//            bd0 = trackBox0.boundingRect();
//            bd1 = trackBox1.boundingRect();
            normalize(post0, post0, 128,255, NORM_MINMAX);
            normalize(post1, post1, 128,255, NORM_MINMAX);
            imshow("afterwards0", post0);
            imshow("afterwards1", post1);
            waitKey(0);
            cout << "-- recalculate histogram" <<endl;
            MaskFromRect(f0_cur, bd0, mask0_motion);
            calcHist(imgSet0, imgCount, channels, mask0_motion, hist0, dims, sizes, ranges);


            MaskFromRect(f1_cur, bd1, mask1_motion);
            calcHist(imgSet1, imgCount, channels, mask1_motion, hist1, dims, sizes, ranges);

            normalize(hist0, hist0, 0,1,NORM_MINMAX);
            normalize(hist1, hist1, 0,1,NORM_MINMAX);

            }


            else
            {
                cout << "no motion detected"<<endl;
            }


            found_box = true;
            break;
        }




        default:
        {
            cout << "other methods are not implemented." <<endl;
            break;

        }

    }

    return found_box;

}








inline void StereoVision::ImagePreprocessing(const cv::Mat& f, cv::Mat& out)
///this function will Gaussian smooth it
{
    float sigma = 1.0;
//    cv::cvtColor(f, out, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(f, out, cv::Size(0,0), sigma,sigma, BORDER_REFLECT);

}


void StereoVision::Tracking3DInitialize(cv::Mat& f1_pre, cv::Mat& f2_pre, cv::Mat& f1_cur, cv::Mat& f2_cur)
{

    /// keypoint detect
    StipDetector detector1 (f1_cur,f1_pre);
    StipDetector detector2 (f2_cur,f2_pre);
    vector<cv::KeyPoint> kpt1;
    vector<cv::KeyPoint> kpt2;
    cv::Mat roi1, roi2;

    detector1.SetMethodROI(StipDetector::TemporalThreshold);
    detector1.detect(StipDetector::FeatureMethod::ORB);
    detector1.GetKeyPoints(kpt1);
//    detector1.GetROI(roi1);
//    detector1.ClearPoints();
//    detector1.VideoKeypointDisplay("frame1");

    detector2.SetMethodROI(StipDetector::TemporalThreshold);
    detector2.detect(StipDetector::FeatureMethod::ORB);
    detector2.GetKeyPoints(kpt2);
//    detector2.ClearPoints();
//    detector2.VideoKeypointDisplay("frame2");
//    detector2.GetROI(roi2);

    /// keypoint description and matching
    cv::Mat description1, description2;

    detector1.GetDescriptorORB(description1);
    detector2.GetDescriptorORB(description2);

    cv::BFMatcher matcher(NORM_HAMMING);
    std::vector<cv::DMatch> matches;

    matcher.match(description1, description2, matches);
    cv::Mat display_match;
    drawMatches( f1_cur, kpt1, f2_cur, kpt2, matches, display_match );
    namedWindow("matches", CV_WINDOW_NORMAL);
    imshow("matches", display_match);




//    cv::imshow("roi1",roi1);
//    cv::imshow("roi2",roi2);


}





StereoVision::~StereoVision()
{
    //dtor
}
