#include "MyHistogram.h"

MyHistogram::MyHistogram()
{
    //ctor
}

MyHistogram::~MyHistogram()
{
    //dtor
}

MyHistogram::MyHistogram(const cv::Mat& image, const cv::Mat& mask, FeatureSpace method)
{
    _img = image.clone();
    _mask = mask.clone();
    _method = method;

}


void MyHistogram::SetImage(const cv::Mat& image)
{
    _img = image.clone();
}


void MyHistogram::SetMask(const cv::Mat& image)
{
    _mask = image.clone();
}

void MyHistogram::GetHist(cv::Mat& out)
{
    out = _hist.clone();
}








void MyHistogram::ComputeHist()
{
    switch(_method)
    {
    case RGB:
        {
            int imgCount = 1;
            int dims = 3;
            const int sizes[] = {256,256,256};
            const int channels[] = {0,1,2};
            float rRange[] = {0,256};
            float gRange[] = {0,256};
            float bRange[] = {0,256};
            const float *ranges[] = {rRange,gRange,bRange};
            calcHist(&_img, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            break;
        }

    case HSV:

        {
            Mat hsv;
            cv::cvtColor(_img, hsv, COLOR_BGR2HSV);
            int imgCount = 1;
            int dims = 3;
            const int sizes[] = {180,256,256};
            const int channels[] = {0,1,2};
            float hRange[] = {0,180};
            float sRange[] = {0,256};
            float vRange[] = {0,256};
            const float *ranges[] = {hRange,sRange,vRange};
            calcHist(&hsv, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            break;
        }


    case LBP:
        /// only single layer uniform local binary pattern, R = 1, P = 8
        {


            Mat img_gray, img_lbp;
            cv::cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            int radius = 1;
            int points = 8;
            LocalBinaryPattern pattern0(radius, points);
            pattern0.UniformLBP(img_gray, img_lbp);

            /// histogram parameters
            int imgCount = 1;
            int dims = 1;
            const int bin0 = pattern0.GetNumBins();
            int range0 = pattern0.GetRanges();

            const int sizes[] = {bin0};
            const int channels[] = {0};
            float lbp0Range[] = {0,range0};
            const float *ranges[] = { lbp0Range};
            calcHist(&img_lbp, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            break;
        }
    default:
        cout << "NOTE: no other features are implemented so far" <<endl;
        break;

    }
}










inline void MyHistogram::SingleBoundingBoxFromROI(cv::Mat& roi, cv::Rect& bd)
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







void MyHistogram::BackProjection( const cv::Mat& in, cv::Mat& backproj)
/// the output backprojection is float type and ranges (0,1)
{

    int nx = in.cols;
    int ny = in.rows;

    backproj = cv::Mat::zeros(in.size(),CV_32F);

    switch(_method)
    {
    case RGB:
        {
            int imgCount = 1;
            int dims = 3;
            const int sizes[] = {256,256,256};
            const int channels[] = {0,1,2};
            float rRange[] = {0,256};
            float gRange[] = {0,256};
            float bRange[] = {0,256};
            const float *ranges[] = {rRange,gRange,bRange};

            cv::calcBackProject(&in,imgCount,channels,_hist,backproj,ranges);

            break;
        }

    case HSV:

        {
            Mat hsv;
            cv::cvtColor(_img, hsv, COLOR_BGR2HSV);
            int imgCount = 1;
            int dims = 3;
            const int sizes[] = {180,256,256};
            const int channels[] = {0,1,2};
            float hRange[] = {0,180};
            float sRange[] = {0,256};
            float vRange[] = {0,256};
            const float *ranges[] = {hRange,sRange,vRange};
            cv::calcBackProject(&in,imgCount,channels,_hist,backproj,ranges);
            break;
        }


    case LBP:
        /// only single layer uniform local binary pattern, R = 1, P = 8
        /// single pixel backprojection does not work here, we use patch-based similarity
        {

            /// encodes to local binary pattern
            Mat img_gray, img_lbp;
            cv::cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            int radius = 1;
            int points = 8;
            LocalBinaryPattern pattern0(radius, points);
            pattern0.UniformLBP(img_gray, img_lbp);

            /// histogram parameters
            int imgCount = 1;
            int dims = 1;
            const int bin0 = pattern0.GetNumBins();
            int range0 = pattern0.GetRanges();

            const int sizes[] = {bin0};
            const int channels[] = {0};
            float lbp0Range[] = {0,range0};
            const float *ranges[] = { lbp0Range};

            /// define search space
            cv::Rect bd0;
            SingleBoundingBoxFromROI(_mask, bd0);
            Point c0 = (bd0.br()+bd0.tl())/2.0;
            const int hx = 20;
            const int hy = 20;

            /// search the nearest 100 patches
            for(int i = -5; i < 5; i++)
            {
                for(int j = -5; j < 5; j++)
                {
                    int x = max( min(c0.x+i*hx, nx-bd0.width/2), bd0.width/2); // the new x should be within the image domain
                    int y = max( min(c0.y+j*hy, ny-bd0.height/2), bd0.height/2); // the new y should be within the image domain

                    Point c (x,y);
                    bd = bd0 + (c-c0);
                    Mat patch = img_lbp(bd);
                    Mat hist;
                    calcHist(&patch, imgCount, channels, cv::Mat(), hist, dims, sizes, ranges);

                    backproj.at<Point>(c) = campareHist(hist,_hist, HISTCMP_KL_DIV );
                }
            }


            break;
        }
    default:
        cout << "NOTE: no other features are implemented so far" <<endl;
        break;

    }
}




















