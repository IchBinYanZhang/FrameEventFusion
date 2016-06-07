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



inline void StereoVision::ShowTracking(const cv::Mat& f_current, cv::Rect& bd, std::vector<cv::Point>& trajectory)
{


        cv::namedWindow("tracking", CV_WINDOW_NORMAL);
        cv::RNG rng(12345);
        cv::Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::rectangle( f_current, bd.tl(), bd.br(), color, 2, 8, 0 );
        for(int i = 0; i < trajectory.size()-1; i++)
            cv::line(f_current, trajectory[i], trajectory[i+1], color);

        /// Show in a window
        cv::imshow("tracking",f_current);

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





void StereoVision::CLBP( Mat& src, Mat& dst, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...

    /// convert to CV_32F first

    src.convertTo(src,CV_32F);
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<float>(i+fy,j+fx) + w2*src.at<float>(i+fy,j+cx) + w3*src.at<float>(i+cy,j+fx) + w4*src.at<float>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<uint8_t>(i,j) += ((t > src.at<float>(i,j)) && (abs(t-src.at<float>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
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


bool StereoVision::Tracking2D( const cv::Mat& f0, const cv::Mat& f1, cv::Rect& bd, cv::Rect& bd_new,cv::Mat& hist, Tracking2DMethod method)
/// f1, f0 the current frame and the previous frame
/// bd, in parameters
/// hist, in
/// bd_new, out the new bounding box
{

    cv::Mat roi;
    bool found_box = false;


    Tracking2DMethod _method = TRACKINGBYDETECTION;
    Mat image, image_h, image_s, image_v,image_lbp, mask = Mat::zeros(f1.size(), CV_8UC1), backproj;

    /// histogram parameters lbp
    int imgCount = 3;
    int dims = 3;
    const int sizes[] = {32,32,32};
    const int channels[] = {0,1,2};
    float sRange[] = {0,256};
    float vRange [] = {0,256};
    float lbpRange[] = {0,256};
    const float *ranges[] = { sRange,vRange, lbpRange};
    float radius = 2;
    int n_points = 8;


    /// histogram parameters of HSV

//    int imgCount = 1;
//    int dims = 2;
//    const int sizes[] = {30,30};
//    const int channels[] = {0,1};
//    float hRange[] = {0,180};
//    float sRange[] = {0,256};
//    const float *ranges[] = {hRange, sRange};
//    float radius = 1;
//    int n_points = 8;

    switch(method)
    {


        case TRACKINGBYDETECTION:
        /// it directly gives us the bounding box. This can be used for tracking initialization.
        {

            StipDetector dec (f0, f1);
            dec.DefineROI();
            dec.GetROI(roi);
            cv::imshow("roi",roi);
            roi.convertTo(roi,CV_8UC1);
            double minval, maxval;
            cv::minMaxLoc(roi,&minval, &maxval);

            if(minval == maxval)
            {
                std::cout << minval << " " <<maxval <<std::endl;
                std::cout << "no motion is detected" <<std::endl;

            }
            else
            {
                /// compute the bounding box
                SingleBoundingBoxFromROI(roi, bd);
                bd_new = bd;
                found_box = true;

                /// compute the histogram

                //CLBP
                cvtColor(f1, image, COLOR_BGR2GRAY);
                CLBP(image, image_lbp, radius, n_points);

                //HSV
                cvtColor(f1,image,COLOR_BGR2HSV);
                int ch1[] = {1, 0};
                int ch2[] = {2, 0};

                image_s.create(image.size(), image.depth());
                mixChannels(&image, 1, &image_s, 1,ch1, 1);
                image_v.create(image.size(), image.depth());
                mixChannels(&image, 1, &image_v, 1, ch2, 1);



                cv::Mat imgSet [] = {image_s,image_v, image_lbp};

                vector<Point> bd_vertices {Point(bd.x, bd.y), Point(bd.x+bd.width, bd.y),Point(bd.x+bd.width, bd.y+bd.height),Point(bd.x, bd.y+bd.height)};
                vector<Point> bd_poly;
                approxPolyDP(bd_vertices, bd_poly, 1.0, true);
                fillConvexPoly(mask, &bd_poly[0], (int)bd_poly.size(), 255, 8, 0);


//                Mat image_lbp_roi (image_lbp, bd);
//                Mat mask_roi(mask, bd);
                calcHist(imgSet, imgCount, channels, mask, hist, dims, sizes, ranges);
//                normalize(hist, hist, 0, 255, NORM_MINMAX);
            }

            break;
        }
        case CAMSHIFT:
        /// update the bounding box using the camshift algorithm, in which the size,location and rotation will update at the same time.
        {


            ///extract histogram
            cvtColor(f1, image, COLOR_BGR2GRAY);
            CLBP(image,image_lbp, radius, n_points);

            cvtColor(f1, image, COLOR_BGR2HSV);
            int ch1[] = {1, 0};
            int ch2[] = {2, 0};
            image_s.create(image.size(), image.depth());
            mixChannels(&image, 1, &image_s, 1,ch1, 1);
            image_v.create(image.size(), image.depth());
            mixChannels(&image, 1, &image_v, 1, ch2, 1);


            cv::Mat imgSet[] = {image_s,image_v, image_lbp};

//
//            vector<Point> bd_vertices {Point(bd.x, bd.y), Point(bd.x+bd.width, bd.y),Point(bd.x+bd.width, bd.y+bd.height),Point(bd.x, bd.y+bd.height)};
//            vector<Point> bd_poly;
//            approxPolyDP(bd_vertices, bd_poly, 1.0, true);
//            fillConvexPoly(mask, &bd_poly[0], (int)bd_poly.size(), 255, 8, 0);


            cv::calcBackProject(imgSet,imgCount,channels,hist,backproj,ranges);
//            cv::GaussianBlur(backproj, backproj, cv::Size(0,0), 6.5,6.5, BORDER_REFLECT);

            imshow("backproj", backproj);
            RotatedRect trackBox = CamShift(backproj, bd,
                                TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));

            bd_new = trackBox.boundingRect();
            /// update histogram
//            vector<Point> bd_vertices1 {Point(bd.x, bd.y), Point(bd.x+bd.width, bd.y),Point(bd.x+bd.width, bd.y+bd.height),Point(bd.x, bd.y+bd.height)};
//            vector<Point> bd_poly1;
//            approxPolyDP(bd_vertices1, bd_poly1, 1.0, true);
//            fillConvexPoly(mask, &bd_poly1[0], (int)bd_poly1.size(), 255, 8, 0);
//
//            calcHist(&image, imgCount, channels, mask, hist, dims, sizes, ranges);

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



void StereoVision::StereoShow(bool is_rectified)
{
    std::cout << " - Frame Stream Display." <<std::endl;
    int num_frames =  this->_stream[0].get(CV_CAP_PROP_FRAME_COUNT);
    cv::namedWindow("stream1", cv::WINDOW_NORMAL);
    cv::namedWindow("stream2", cv::WINDOW_NORMAL);
    cv::Mat frame1, frame1_rec, frame1_pre, frame1_s;
    cv::Mat frame2, frame2_rec, frame2_pre, frame2_birdview, frame2_s;
    std::vector<cv::Point> trajectory;
    /// params for rectification
    cv::Mat R1, R2, P1, P2, Q;

    //Create transformation and rectification maps
    cv::Mat cam1map1, cam1map2;
    cv::Mat cam2map1, cam2map2;

    /// for tracking
    cv::Rect bd, bd_new;
    cv::Mat hist;



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
        trajectory.clear();

        for(int i = 0; i < num_frames; i++)
        {

            this->_stream[0].read(frame1);
            this->_stream[1].read(frame2);


            cout << "------------frame "<<i<<" -----------------"<<endl;

            cv::imshow("stream1",frame1);
            cv::imshow("stream2",frame2);
            cv::namedWindow("birdview", CV_WINDOW_NORMAL);
            ImagePreprocessing(frame2, frame2_s); // only gaussian blur

            if(i>50  )
            {
                if(bd.height == 0.0 || bd.width == 0.0){
                    cout << " -- detection" <<endl;
                    Tracking2D(frame2_pre, frame2_s, bd, bd_new, hist, TRACKINGBYDETECTION);
                }
                else
                {
                    cout << " -- tracking" <<endl;
                    Tracking2D(frame2_pre, frame2_s, bd, bd_new, hist,CAMSHIFT);
                    trajectory.push_back( (bd_new.tl()+bd_new.br())/2 );

                    ShowTracking(frame2,bd_new, trajectory);
                    cv::warpPerspective(frame2, frame2_birdview, _H, cv::Size(frame1.cols, frame1.rows));
                    cv::imshow("birdview", frame2_birdview);
//                    cv::waitKey(0);
                }
            }


            frame2_pre = frame2_s.clone();

            cv::waitKey(10);
        }

    }


    std::cout << " - Frame Stream Display Ends." <<std::endl;

}







inline void StereoVision::ImagePreprocessing(const cv::Mat& f, cv::Mat& out)
///this function will convert the image to gray value and Gaussian smooth it
{
    float sigma = 0.5;
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
