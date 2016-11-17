#include <opencv2/imgproc.hpp>
#include "StereoVision.h"

using namespace std;
using namespace cv;
//using namespace cv::imgproc;

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


//void StereoVision::HomographyToGround(cv::Mat& img_cb, bool img_show)
///// find the homography, given the image of a checkerboard on the ground, with 5 corners per row and 7 corners per column on the pattern
//{
//    int nx = 5;
//    int ny = 7;
//    cv::Size pattern_size(nx,ny);
//    std::vector<cv::Point2f> corners;
//    std::vector<cv::Point2f> corners_w;
//    bool patternfound = cv::findChessboardCorners(img_cb, pattern_size, corners,
//        CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
//        + CALIB_CB_FAST_CHECK);
//
//    for(auto i = corners.begin(); i < corners.end(); i++)
//    {
//        double x;
//        double y;
//        int N = std::distance(corners.begin(), i);
//        x = N%nx;
//        y = (N-x)/nx;
//        corners_w.push_back(cv::Point2f(5*x+img_cb.cols/4,5*y+3*img_cb.rows/4));
//
//    }
//
//    _H = cv::findHomography(corners,corners_w);
//
//
//    if(img_show)
//    {
//
//        cv::drawChessboardCorners(img_cb, pattern_size, Mat(corners), patternfound);
//        cv::namedWindow("corners",WINDOW_NORMAL);
//        imshow("corners", img_cb);
//        cv::waitKey(5);
//    }
//}



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


    /// compute projection matrices
    Mat temp;
    for(int i = 0; i < _intrisic_mat.size(); i++){
        hconcat(_rotation_mat[i], _trans_vec[i], temp);
        _projection_mat.push_back(_intrisic_mat[i]*temp);

    }
    _projection_mat_rec.push_back(P1);
    _projection_mat_rec.push_back(P2);

}




void StereoVision::SetGroundPlane(cv::Mat& plane)
{
    _ground_plane = plane.clone();
}


void StereoVision::ProjectWorldPointsToGround(std::vector<cv::Mat>& src,std::vector<cv::Point2f>& dst)
/// input src is in homogeneous coordinate
{
    int N = src.size();
    for(int i = 0; i < N; i++){
        float x = _ground_plane.at<double>(0,0) * src[i].at<float>(0)/src[i].at<float>(3) + _ground_plane.at<double>(1,0) * src[i].at<float>(1)/src[i].at<float>(3)
                + _ground_plane.at<double>(2,0) * src[i].at<float>(2)/src[i].at<float>(3);
        float y = _ground_plane.at<double>(0,1) * src[i].at<float>(0)/src[i].at<float>(3) + _ground_plane.at<double>(1,1) * src[i].at<float>(1)/src[i].at<float>(3)
                + _ground_plane.at<double>(2,1) * src[i].at<float>(2)/src[i].at<float>(3);
        dst.push_back(Point2f(x,y));
    }
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

inline void StereoVision::ShowTracking(std::vector<cv::Point2f>& trajectory, cv::Mat& out)
{

        out = Mat::zeros(_ny, _nx, CV_32F);
//        cv::RNG rng(12345);
        cv::Scalar color = Scalar( 36, 150, 223 );
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
/// F is float type
{

    /// move the origin so that P = K[I|0] and P' = K[_R|_T]
    cv::Mat _R, _T, _T_cross; // relative transform from C to C', so that P = K[I|0] and P' = K[_R|_T]
    _R = R_prime * R.inv();
    _T = T_prime - T;
    _T_cross = (cv::Mat_<double>(3,3)<< 0, -_T.at<double>(2), _T.at<double>(1), _T.at<double>(2), 0, -_T.at<double>(0), -_T.at<double>(1),_T.at<double>(0),0);

    F = (K_prime.inv()).t() * _T_cross * _R * K.inv();
    F.convertTo(F, CV_32F);

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







inline void EpanechnikovKernel(const Mat& image, Rect& bbox, float c, Mat& kernel)
{

    Point x0 = (bbox.tl()+bbox.br())/2.0;
    for(int j = 0; j < kernel.rows; j++){
            float* ptr = kernel.ptr<float>(j);
        for(int i = 0; i < kernel.cols; i++){
            float dist2 = ((float)i-x0.x)*((float)i-x0.x) + ((float)j-x0.y)*((float)j-x0.y);
            ptr[i] = (dist2 <=c*c)? 1.0f-dist2/(c*c) : 0.0f;
        }
    }

}


//
bool StereoVision::Tracking2D( const cv::Mat& f0, const cv::Mat& f1, cv::Rect& bd,cv::Mat& hist, Tracking2DMethod method)
/// f1, f0 the current frame and the previous frame
/// bd, in parameters
/// hist, in
/// bd_new, out the new bounding box
{

    cv::Mat roi;
    bool found_box = false;


    Tracking2DMethod _method = TRACKINGBYDETECTION;
    Mat image, image_h, image_s, image_v,image_lbp, mask = Mat::zeros(f1.size(), CV_8UC1);
    cv::Mat kernel = Mat::zeros(f1.size(), CV_32F);



    switch(method)
    {


        case TRACKINGBYDETECTION:
        /// it directly gives us the bounding box. This can be used for tracking initialization.
        {
            cv::Rect bd_tmp;
            cv::Mat mask;
            StipDetector dec (f0, f1);
            dec.DefineROI();
            dec.GetROI(roi);
            roi.convertTo(roi,CV_8UC1);
            SingleBoundingBoxFromROI(roi, bd_tmp);



            if(bd_tmp.area() <= 5000)
            {
                std::cout << " -- no motion is detected." <<std::endl;
            }
            else
            {
                /// compute the bounding box
                bd = bd_tmp;
                found_box = true;


                /// compute the histogram
                MyHistogram hist_generator(f1, bd, MyHistogram::HSV);
                EpanechnikovKernel(f1, bd, 11.0, kernel);
                hist_generator.ComputeHistWeights(kernel);
                hist_generator.GetHist(hist);
            }

            break;
        }
        case CAMSHIFT:
        /// update the bounding box using the camshift algorithm, in which the size,location and rotation will update at the same time.
        {

            /// extract motion
            cv::Rect bd_motion;
            StipDetector dec (f0, f1);
            dec.DefineROI();
            dec.GetROI(roi);
            roi.convertTo(roi,CV_8UC1);
            SingleBoundingBoxFromROI(roi, bd_motion);
            if(bd_motion.area() <= bd.area())
                bd_motion = bd;

            ///extract histogram
            Mat hist_candidate, backproj_candidate;
            Mat hist_template, backproj_template;
            Mat backproj;
            MyHistogram hist_generator(f1, bd, MyHistogram::HSV);
            EpanechnikovKernel(f1, bd, 11.0, kernel);
            hist_generator.ComputeHistWeights(kernel);
            hist_generator.GetHist(hist_candidate);
            hist_generator.BackProjection(f1,backproj_candidate);
            hist_generator.SetHist(hist);
            hist_generator.BackProjection(f1,backproj_template);
            cv::divide(backproj_template, backproj_candidate,backproj);
            cv::sqrt(backproj, backproj);

            cv::GaussianBlur(backproj, backproj, cv::Size(0,0), 2.5,2.5, BORDER_REFLECT);

            cv::Mat mask_motion;
            MaskFromRect(f1, bd_motion, mask_motion);
            cv::distanceTransform(mask_motion, mask_motion,CV_DIST_L2,CV_DIST_MASK_PRECISE);
            mask_motion.convertTo(mask_motion,CV_32F, 1.0f/255);

            backproj = backproj.mul(mask_motion);
//            RotatedRect trackBox = CamShift(backproj, bd,
//                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 5, 1 ));
            meanShift(backproj,bd, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 5, 1 ));

            normalize(backproj, backproj, 0,1, NORM_MINMAX);
            //imshow("backproj", backproj);

            ///update bounding box and template histgram
//            bd = trackBox.boundingRect();
            hist_generator.SetBoundingBox(bd);
            EpanechnikovKernel(f1, bd, 11.0, kernel);

            hist_generator.ComputeHistWeights(kernel);
            hist_generator.GetHist(hist);

            found_box = true;
            break;
        }


        default:
            break;
    }

//    HistDisplay(hist, sizes[0], ranges[0] );
    return found_box;

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




//void StereoVision::BlockMatching2(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& frame1_roi, cv::Mat& out)
//{
//    /// Image domain Omega
//    int nx = frame1.cols;
//    int ny = frame1.rows;
//    cv::Mat segments;
//    cv::Mat segments2;
//
//    cv::Ptr<SuperpixelSLIC> su = cv::ximgproc::createSuperpixelSLIC(frame1, SLICO,20, 15);
//    cv::Ptr<SuperpixelSLIC> su2 = cv::ximgproc::createSuperpixelSLIC(frame2, SLICO,20, 15);
//
//    su->iterate(10);
//    su->enforceLabelConnectivity(25);
//
//    su->getLabelContourMask(segments,false);
//    frame1.setTo(Scalar(255),segments);
//
//    su2->iterate(10);
//    su2->enforceLabelConnectivity(25);
//
//    su2->getLabelContourMask(segments2,false);
//    frame2.setTo(Scalar(255),segments2);
//
//
//    cv::imshow("pixels", frame1);
//    cv::imshow("pixels2", frame2);
//
//        std::cout <<"matching ends"<<std::endl;
//
//    cv::waitKey(10);
//
//}



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


inline void StereoVision::UpdateGeometryByImageResize( float fx, float fy)
/// image is resized by factor fx in x-direction and fy in y-direction
{
    cv::Mat factor_mat = (Mat_<double>(3,3) << fx, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 1.0);
    for(int i = 0; i < _intrisic_mat.size(); i++){
        _intrisic_mat[i] = factor_mat*_intrisic_mat[i];
        _projection_mat[i] = factor_mat*_projection_mat[i];

    }

    _nx = fx*_nx;
    _ny = fy*_ny;



    /// computings for image rectification transformations
    _projection_mat_rec.clear();
    cv::Mat R1, P1, R2, P2, Q;


    cv::stereoRectify(_intrisic_mat[0],_dist_coeff[0],_intrisic_mat[1],_dist_coeff[1],
                      cv::Size(_nx,_ny), _rotation_mat[1],_trans_vec[1],
                      R1,R2,P1,P2,Q,CV_CALIB_ZERO_DISPARITY,1, cv::Size(_nx,_ny));


    cv::initUndistortRectifyMap(_intrisic_mat[0], _dist_coeff[0], R1, P1, cv::Size(_nx,_ny),
                                 CV_16SC2, _cam1map1, _cam1map2);
    cv::initUndistortRectifyMap(_intrisic_mat[1], _dist_coeff[1], R2, P2, cv::Size(_nx,_ny),
                                     CV_16SC2, _cam2map1, _cam2map2);

    _projection_mat_rec.push_back(P1);
    _projection_mat_rec.push_back(P2);


}




void StereoVision::StereoShow(bool is_rectified, const string& filename_traj)
{
    std::cout << " - Frame Stream Display." <<std::endl;
    int num_frames =  this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
//    cv::namedWindow("stream1", cv::WINDOW_NORMAL);
//    cv::namedWindow("stream2", cv::WINDOW_NORMAL);
    cv::Mat frame1, frame1_ud,frame1_rec, frame1_pre, frame1_birdview, frame1_s, frame1_tracking,frame1_tracking_est;
    cv::Mat frame2, frame2_ud,frame2_rec, frame2_pre, frame2_birdview, frame2_s, frame2_tracking, frame2_tracking_est;
    std::vector<cv::Point2f> trajectory1;
    std::vector<cv::Point2f> trajectory1_estimated;
    std::vector<cv::Point2f> trajectory2;
    std::vector<cv::Point2f> trajectory2_estimated;
    std::vector<Mat> trajectory3D_homo;
    std::vector<Mat> trajectory3D_euc;
    std::vector<Point2f> trajectory_bird;
    Mat P0;
    Mat x0;

    std::fstream traj_3Dhomo;

    /// params for rectification
    cv::Mat R1, R2, P1, P2, Q;

    //Create transformation and rectification maps
    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    /// for tracking
    cv::Rect bd1, bd2;
    cv::Mat hist1, hist2;
    cv::Mat x_pre, x_cur; // 3D points

    double resize_factor_x = 0.5, resize_factor_y = 0.5;
    UpdateGeometryByImageResize(resize_factor_x, resize_factor_y);

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
        trajectory1_estimated.clear();
        trajectory2.clear();
        trajectory2_estimated.clear();

        for(int i = 0; i < num_frames; i++)
        {

            this->_stream[0].read(frame1);
            this->_stream[1].read(frame2);

            if(frame2.empty() | frame1.empty())
            {
                cout<< "----------Display Ends--------------"<<endl;
            }
            else{
                cout << "------------frame "<<i<<" -----------------"<<endl;
                resize(frame2,frame2,Size(0,0), resize_factor_x, resize_factor_y);
                resize(frame1,frame1,Size(0,0), resize_factor_x, resize_factor_y);

                ImagePreprocessing(frame1, frame1_s); // only gaussian blur
                ImagePreprocessing(frame2, frame2_s); // only gaussian blur

                if(i>100)
                {
                    if(bd1.height == 0.0 || bd1.width == 0.0 || bd2.height == 0.0 || bd2.width == 0.0)
                    {

                        P0 = Mat::zeros(7,7,CV_32F);
                        cout << " -- detection" <<endl;
//                        Tracking3D(frame1_pre, frame1_s, bd1, hist1, frame2_pre, frame2_s, bd2, hist2, TRACKINGBYDETECTION3, HSVLBP);
                        if(Tracking2D( frame2_pre, frame2_s, bd2,hist2, TRACKINGBYDETECTION) && Tracking2D( frame1_pre, frame1_s, bd1,hist1, TRACKINGBYDETECTION)){
                            Tracking3DInitialize( frame1_pre, frame1_s, bd1, frame2_pre, frame2_s, bd2,x0);
                            trajectory3D_homo.push_back(x0);

                            setIdentity(P0, Scalar::all(0.1));
                        }
                    }
                    else
                    {
                        cout << " -- tracking" <<endl;
                        float vx, vy, vz;
                        Point2f A, B;
                        vector<Mat>::iterator ptr =trajectory3D_homo.end()-1;
                        Mat xt_1 = *(ptr);
                        Mat xt_2;
                        Mat xt;
                        if(trajectory3D_homo.size()>1){
                            xt_2 = *(ptr-1);
                            vx = xt_1.at<float>(0)-xt_2.at<float>(0) ;
                            vy = xt_1.at<float>(1)-xt_2.at<float>(1) ;
                            vz = xt_1.at<float>(2)-xt_2.at<float>(2) ;
                        }
                        else{
                            vx = 0.0f;
                            vy = 0.0f;
                            vz = 0.0f;
                        }

//                        namedWindow("stream1_tracking", CV_WINDOW_NORMAL);
//                        namedWindow("stream2_tracking", CV_WINDOW_NORMAL);
//                        moveWindow("stream1_tracking", 0,0);
//                        moveWindow("stream2_tracking", 0,700);


                        Tracking2D( frame2_pre, frame2_s, bd2, hist2, CAMSHIFT);
                        Tracking2D( frame1_pre, frame1_s, bd1, hist1, CAMSHIFT);
                        Tracking3DKalman(bd1,bd2, xt_1, vx, vy, vz, xt, P0, A, B);

                        trajectory3D_homo.push_back(xt);
                        traj_3Dhomo.open(filename_traj.c_str(), std::fstream::out | std::fstream::app);
                        traj_3Dhomo << xt.at<float>(0)<<"\t"<<xt.at<float>(1)<<"\t"<<xt.at<float>(2)<<"\t"<<xt.at<float>(3)<<"\n";
                        traj_3Dhomo.close();
//                        // project xt to the ground plane
//                        float x = _ground_plane.at<double>(0,0) * (xt.at<float>(0)/xt.at<float>(3)-x0.at<float>(0)/x0.at<float>(3) )
//                                + _ground_plane.at<double>(1,0) * (xt.at<float>(1)/xt.at<float>(3)-x0.at<float>(1)/x0.at<float>(3) )
//                                + _ground_plane.at<double>(2,0) * (xt.at<float>(2)/xt.at<float>(3)-x0.at<float>(2)/x0.at<float>(3) );
//                        float y = _ground_plane.at<double>(0,1) * (xt.at<float>(0)/xt.at<float>(3)-x0.at<float>(0)/x0.at<float>(3) )
//                                + _ground_plane.at<double>(1,1) * (xt.at<float>(1)/xt.at<float>(3)-x0.at<float>(1)/x0.at<float>(3) )
//                                + _ground_plane.at<double>(2,1) * (xt.at<float>(2)/xt.at<float>(3)-x0.at<float>(2)/x0.at<float>(3) );
//
//                        trajectory_bird.push_back(Point2f(x,y));
//                        ShowTracking(trajectory_bird, frame1_birdview);
//                        cout << Point2f(x,y) <<endl;

						/// store the trajs and visualization
                        trajectory1_estimated.push_back(A);
                        ShowTracking(frame1,bd1, trajectory1_estimated, frame1_tracking_est);

                        trajectory2_estimated.push_back(B);
                        ShowTracking(frame2,bd2, trajectory2_estimated, frame2_tracking_est);

                        trajectory1.push_back( Point2f((bd1.tl()+bd1.br())/2) );
                        ShowTracking(frame1,bd1, trajectory1, frame1_tracking);

                        trajectory2.push_back( Point2f((bd2.tl()+bd2.br())/2) );
                        ShowTracking(frame2,bd2, trajectory2, frame2_tracking);

						/// update the bounding box based on point A and B, the 2D projections of xt at cam1 and cam2.
						//Point2f dd1 = A - (Point2f(bd1.br()) + Point2f(bd1.tl())) / 2.0f;
						//Point2f dd2 = B - (Point2f(bd2.br()) + Point2f(bd2.tl())) / 2.0f;
						//bd1 = bd1 + Point(dd1);
						//bd2 = bd2 + Point(dd2);
						
                        cv::imshow("stream1_observation", frame1_tracking);
                        cv::imshow("stream2_observation", frame2_tracking);
                        cv::imshow("stream1_estimated", frame1_tracking_est);
						cv::imshow("stream2_estimated", frame2_tracking_est);
////                        cv::imshow("bird_view", frame1_birdvie)w;
						cv::waitKey(5);
                    }
                }

                frame1_pre = frame1_s.clone();
                frame2_pre = frame2_s.clone();

            }
        }

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

    switch(method)
    {

         case TRACKINGBYDETECTION3:
        /// it directly gives us the bounding box. This can be used for tracking initialization.
        {
            cv::Rect bd0_tmp, bd1_tmp;
            cv::Mat mask0, mask1;
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



            if(bd0_tmp.area() <= 5000 || bd1_tmp.area() <= 5000)
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

                MyHistogram hist_generator0(f0_cur, bd0, MyHistogram::HSVLBP);
                MyHistogram hist_generator1(f1_cur, bd1, MyHistogram::HSVLBP);

                hist_generator0.ComputeHist();
                hist_generator1.ComputeHist();
                hist_generator0.GetHist(hist0);
                hist_generator1.GetHist(hist1);

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
                MyHistogram hist_generator0(f0_cur, bd0, MyHistogram::HSVLBP);
                MyHistogram hist_generator1(f1_cur, bd1, MyHistogram::HSVLBP);
                cout << "1"<<endl;
                hist_generator0.SetHist(hist0);
                hist_generator1.SetHist(hist1);
                cout << "2"<<endl;
                hist_generator0.BackProjection(f0_cur, backproj0);
                hist_generator1.BackProjection(f1_cur, backproj1);
                cout << "3"<<endl;

            cv::GaussianBlur(backproj0, backproj0, cv::Size(0,0), 1.0,1.0, BORDER_REFLECT);
            cv::GaussianBlur(backproj1, backproj1, cv::Size(0,0), 1.0,1.0, BORDER_REFLECT);

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
            for(int k = 0; k < 1; k++)
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

                cv::GaussianBlur(pdf_epi0, pdf_epi0, cv::Size(0,0), 0.5,18.5, BORDER_REFLECT);
                cv::GaussianBlur(pdf_epi1, pdf_epi1, cv::Size(0,0), 0.5,18.5, BORDER_REFLECT);



                imshow("epipolar0", pdf_epi0);
                imshow("epipolar1", pdf_epi1);
//
//                waitKey(5);
                post0 = backproj0.mul(pdf_epi0);
                post1 = backproj1.mul(pdf_epi1);


                post0.convertTo(post0,CV_8UC1, 255);
                post1.convertTo(post1,CV_8UC1, 255);


//                MaskFromRect(f0_cur, bd0_motion, mask0_motion);
//                MaskFromRect(f1_cur, bd1_motion, mask1_motion);
//                post0 &= mask0_motion;
//                post1 &= mask1_motion;


                /// meanshift update
                meanShift(backproj0,bd0, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                meanShift(backproj1,bd1, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));


            }
//            RotatedRect trackBox0 = CamShift(backproj0, bd0_motion,
//                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
//
//            RotatedRect trackBox1 = CamShift(backproj1, bd1_motion,
//                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));


//            bd0 = trackBox0.boundingRect();
//            bd1 = trackBox1.boundingRect();
            normalize(post0, post0, 128,255, NORM_MINMAX);
            normalize(post1, post1, 128,255, NORM_MINMAX);
            imshow("afterwards0", post0);
            imshow("afterwards1", post1);
//            waitKey(0);


            cout << "-- recalculate histogram" <<endl;

            hist_generator0.SetBoundingBox(bd0);
            hist_generator1.SetBoundingBox(bd1);
            hist_generator0.ComputeHist();
            hist_generator1.ComputeHist();
            hist_generator0.GetHist(hist0);
            hist_generator1.GetHist(hist1);

//            cout << hist0 <<endl;

//            MaskFromRect(f0_cur, bd0, mask0_motion);
//            calcHist(imgSet0, imgCount, channels, mask0_motion, hist0, dims, sizes, ranges);
//
//
//            MaskFromRect(f1_cur, bd1, mask1_motion);
//            calcHist(imgSet1, imgCount, channels, mask1_motion, hist1, dims, sizes, ranges);

//            normalize(hist0, hist0, 0,1,NORM_MINMAX);
//            normalize(hist1, hist1, 0,1,NORM_MINMAX);

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


void StereoVision::Tracking3DKalman(Rect& bd0, Rect& bd1, Mat& pt_in, float vx, float vy, float vz, Mat& pt_out, Mat& Pt, Point2f& post0, Point2f& post1)
{

    /// define the standard Kalman filter
    const int dim_state=7;
    const int dim_measure = 6;
    const int dim_control = 0;
    cv::KalmanFilter kf(dim_state, dim_measure, dim_control, CV_32F);
    const float alpha = 0.0;

    // process model
	double lambda = 1;
 //   kf.transitionMatrix = (Mat_<float>(dim_state, dim_state)<<

 //                          1-lambda,0,0,0,lambda,0,0,
 //                          0,1-lambda,0,0,0, lambda,0,
 //                          0,0,1-lambda,0,0,0, lambda,
 //                          0,0,0,1 - lambda,0,0,0,
 //                          0,0,0,0,1,0,0,
 //                          0,0,0,0,0,1,0,
 //                          0,0,0,0,0,0,1
 //                          );

	kf.transitionMatrix = (Mat_<float>(dim_state, dim_state) <<

		1, 0, 0, 0, lambda, 0, 0,
		0, 1, 0, 0, 0, lambda, 0,
		0, 0, 1, 0, 0, 0, lambda,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1
		);




    setIdentity(kf.processNoiseCov, Scalar::all(0.1));
//    kf.processNoiseCov.at<float>(0)= 10.0f;
//    kf.processNoiseCov.at<float>(8)= 10.0f;
//    kf.processNoiseCov.at<float>(24)= 0.0f;

    // observation model
    Mat R = (Mat_<float>(4,dim_state) <<
                       1,0,0,0,0,0,0,
                       0,1,0,0,0,0,0,
                       0,0,1,0,0,0,0,
                       0,0,0,1,0,0,0
                    );

    Mat P0_rec, P1_rec, P0, P1;
//    _projection_mat_rec[0].convertTo(P0_rec, CV_32F);
//    _projection_mat_rec[1].convertTo(P1_rec, CV_32F);
    _projection_mat[0].convertTo(P0, CV_32F);
    _projection_mat[1].convertTo(P1, CV_32F);


    vconcat(P0*R/pt_in.at<float>(2), P1*R/pt_in.at<float>(2), kf.measurementMatrix);
    setIdentity(kf.measurementNoiseCov, Scalar::all(0.1));
 //   kf.measurementNoiseCov.at<float>(14)= 10;
	//kf.measurementNoiseCov.at<float>(35) = 10;



    // set the initial state
    kf.statePost.at<float>(0) = pt_in.at<float>(0);
    kf.statePost.at<float>(1) = pt_in.at<float>(1);
    kf.statePost.at<float>(2) = pt_in.at<float>(2);
    kf.statePost.at<float>(3) = pt_in.at<float>(3);
    kf.statePost.at<float>(4) = vx;
    kf.statePost.at<float>(5) = vy;
    kf.statePost.at<float>(6) = vz;
    kf.errorCovPost = Pt;

    /// find the current observation
    vector<Point2f> points0;
    vector<Point2f> points1;

    Mat ot, x0, x1;
    x0 = (Mat_<float>(3,1) << (bd0.br().x + bd0.tl().x)/2.0f , (bd0.br().y + bd0.tl().y)/2.0f, 1.0f);
    x1 = (Mat_<float>(3,1) << (bd1.br().x + bd1.tl().x)/2.0f , (bd1.br().y + bd1.tl().y)/2.0f, 1.0f);
//    points0.push_back( static_cast<Point2f>((bd0.br()+bd0.tl())/2.0 ) );
//    points1.push_back( static_cast<Point2f>((bd1.br()+bd1.tl())/2.0 ) );



    // map to undistorted image
	//Mat xx0(1, 1, CV_32FC2, Scalar(x0.at<float>(0), x0.at<float>(1)));
	//Mat xx1(1, 1, CV_32FC2, Scalar(x1.at<float>(0), x1.at<float>(1)));
	//Mat points_undist0 = xx0.clone();
	//Mat points_undist1 = xx1.clone();

 //   undistortPoints(xx0, points_undist0,_intrisic_mat[0], _dist_coeff[0], cv::noArray(), _intrisic_mat[0]);
 //   undistortPoints(xx1, points_undist1,_intrisic_mat[1], _dist_coeff[1], cv::noArray(), _intrisic_mat[1]);
	//cout << points_undist0 << endl;
	//cout << points_undist1 << endl;
	//Mat xx0_ud = (Mat_<float>(3, 1) << points_undist0.at<float>(0), points_undist0.at<float>(1), 1.0f);
	//Mat xx1_ud = (Mat_<float>(3, 1) << points_undist1.at<float>(0), points_undist1.at<float>(1), 1.0f);
    vconcat(x0, x1, ot);
	
    cout << "===observation:==="<<endl;
    cout << ot <<endl;

    cout <<"===previous state==="<<endl;
    cout << kf.statePost<<endl;

    /// perform predicting and updating
    Mat predicted = kf.predict();
    Mat estimated = kf.correct(ot);

    pt_out = estimated.rowRange(Range(0,4));
    Pt = kf.errorCovPost;

    cout <<"===prediction" << endl;
    cout <<kf.statePre<<endl;
    cout <<"===update===" << endl;
    cout << estimated<<endl;
//        cout << "estimated covariance"<<kf.errorCovPost<<endl;


    /// project estimated to 2D image planes

    Mat pp0 = P0*pt_out;
    post0 = Point2f(pp0.at<float>(0)/pp0.at<float>(2),pp0.at<float>(1)/pp0.at<float>(2));
    Mat pp1 = P1*pt_out;
    post1 = Point2f(pp1.at<float>(0)/pp1.at<float>(2),pp1.at<float>(1)/pp1.at<float>(2));


    cout <<"== 3D-2D projection on image 1:"<<post0<<endl;
//
    cout <<"== 3D-2D projection on image 2:"<< post1 <<endl;
//    waitKey(0);

}





inline void StereoVision::ImagePreprocessing(const cv::Mat& f, cv::Mat& out)
///this function will Gaussian smooth it
{
    float sigma = 0.5;
//    cv::cvtColor(f, out, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(f, out, cv::Size(0,0), sigma,sigma, BORDER_REFLECT);

}


void StereoVision::Tracking3DInitialize( cv::Mat& f1_pre, cv::Mat& f1_cur, cv::Rect& bd1,
                                         cv::Mat& f2_pre, cv::Mat& f2_cur, cv::Rect& bd2,
                                         cv::Mat& center)
/// this function finds one 3D point (x,y,z,1) in homogeneous space
{



    /// keypoint detect
    StipDetector detector1 (f1_cur,f1_pre);
    StipDetector detector2 (f2_cur,f2_pre);
    vector<cv::KeyPoint> kpt1;
    vector<cv::KeyPoint> kpt2;
    cv::Mat roi1, roi2;

    detector1.SetMethodROI(StipDetector::Manual);
    detector1.SetBoundingBox(bd1);
    detector1.detect(StipDetector::FeatureMethod::ORB);
    detector1.GetKeyPoints(kpt1);


    detector2.SetMethodROI(StipDetector::Manual);
    detector2.SetBoundingBox(bd2);
    detector2.detect(StipDetector::FeatureMethod::ORB);
    detector2.GetKeyPoints(kpt2);


    /// keypoint description and matching
    cv::Mat description1, description2;

    detector1.GetDescriptorORB(description1);
    detector2.GetDescriptorORB(description2);

    cv::BFMatcher matcher(NORM_HAMMING);
    std::vector<cv::DMatch> matches_tmp;
    std::vector<cv::DMatch> matches;


    matcher.match(description1, description2, matches_tmp);




    /// remove outliers by fundamental matrix
    Mat F;
    vector<Point2f> pt_set1;
    vector<Point2f> pt_set2;
    FundamentalMatrixFromCalibration(_intrisic_mat[0], _rotation_mat[0], _trans_vec[0],
                                     _intrisic_mat[1], _rotation_mat[1], _trans_vec[1],
                                     F);
	cout << F << endl;
    for(int i = 0; i < matches_tmp.size(); i++){
        Point2f x = kpt1[matches_tmp[i].queryIdx].pt;
        Point2f y = kpt2[matches_tmp[i].trainIdx].pt;

        // compute | <x', Fx> |
        float innerF = fabs( (F.at<float>(0,0)* x.x + F.at<float>(0,1)* x.y + F.at<float>(0,2))*y.x
                     + (F.at<float>(1,0)* x.x + F.at<float>(1,1)* x.y + F.at<float>(1,2))*y.y
                     + (F.at<float>(2,0)* x.x + F.at<float>(2,1)* x.y + F.at<float>(2,2)) );

        if(innerF < 1.5f){
            matches.push_back(matches_tmp[i]);
            pt_set1.push_back(x);
            pt_set2.push_back(y);
        }
    }
    matches_tmp.clear();

    /// visualize the matching
    //cv::Mat display_match;
    //drawMatches( f1_cur, kpt1, f2_cur, kpt2, matches, display_match );
    //namedWindow("matches", CV_WINDOW_NORMAL);
    //imshow("matches", display_match);
    //waitKey(0);

    /// triangulate points

    Mat pts3D_homo;
    triangulatePoints(_projection_mat[0], _projection_mat[1],pt_set1, pt_set2, pts3D_homo);
        // from homogeneous space to euclidean space
    Mat scale = pts3D_homo.row(3);
    Mat temp;
    repeat(scale, 4,1,temp);
    pts3D_homo = pts3D_homo / temp;
        // read out the center of the 3D point clouds
    reduce(pts3D_homo,center, 1, CV_REDUCE_AVG);


    //cv::imshow("roi1",roi1);
    //cv::imshow("roi2",roi2);


}





StereoVision::~StereoVision()
{
    //dtor
}
