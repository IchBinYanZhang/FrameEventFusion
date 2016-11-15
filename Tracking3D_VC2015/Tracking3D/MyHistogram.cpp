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


MyHistogram::MyHistogram(const cv::Mat& image, const cv::Rect& bd0, FeatureSpace method)
{
    _img = image.clone();
    _method = method;
    _mask = cv::Mat::zeros(image.size(), CV_8UC1);
    vector<Point> bd_vertices0 {Point(bd0.x, bd0.y), Point(bd0.x+bd0.width, bd0.y),Point(bd0.x+bd0.width, bd0.y+bd0.height),Point(bd0.x, bd0.y+bd0.height)};
    vector<Point> bd_poly0;
    approxPolyDP(bd_vertices0, bd_poly0, 1.0, true);
    fillConvexPoly(_mask, &bd_poly0[0], (int)bd_poly0.size(), 255, 8, 0);
}










void MyHistogram::SetImage(const cv::Mat& image)
{
    _img = image.clone();
}


void MyHistogram::SetMask(const cv::Mat& image)
{
    _mask = image.clone();
}


void MyHistogram::SetBoundingBox(const cv::Rect& bd0)
{
    _mask = cv::Mat::zeros(_img.size(), CV_8UC1);
    vector<Point> bd_vertices0 {Point(bd0.x, bd0.y), Point(bd0.x+bd0.width, bd0.y),Point(bd0.x+bd0.width, bd0.y+bd0.height),Point(bd0.x, bd0.y+bd0.height)};
    vector<Point> bd_poly0;
    approxPolyDP(bd_vertices0, bd_poly0, 1.0, true);
    fillConvexPoly(_mask, &bd_poly0[0], (int)bd_poly0.size(), 255, 8, 0);
}







void MyHistogram::GetHist(cv::Mat& out)
{
    out = _hist.clone();
}


void MyHistogram::SetHist(const cv::Mat& hist)
{
    _hist = hist.clone();
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
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(_img, image0, COLOR_BGR2HSV);
            int imgCount = 3;
            int dims = 3;
            const int sizes[] = {45,64,64};
            const int channels[] = {0,1,2};
            float hRange[] = {0,1};
            float sRange[] = {0,1};
            float vRange[] = {0,1};
            const float *ranges[] = {hRange,sRange,vRange};


            split(image0, splitHSV);
            splitHSV[0].convertTo(splitHSV[0], CV_32F, 1.0f/180);
            splitHSV[1].convertTo(splitHSV[1], CV_32F, 1.0f/256);
            splitHSV[2].convertTo(splitHSV[2], CV_32F, 1.0f/256);


            Mat imageSet[] = {splitHSV[0],splitHSV[1], splitHSV[2]};

            calcHist(imageSet, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            break;
        }


    case LBP:
        /// only single layer uniform local binary pattern, R = 1, P = 8
        {


            Mat img_gray, img_lbp;
            cv::cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            int radius = 1;
            int points = 8;
            LocalBinaryPattern pattern0(radius, points, false);
            pattern0.UniformLBP(img_gray, img_lbp);
            img_lbp.convertTo(img_lbp, CV_32F, 1.0/255);

            /// histogram parameters
            int imgCount = 1;
            int dims = 1;
            const int bin0 = pattern0.GetNumBins();
            int range0 = pattern0.GetRanges();

            const int sizes[] = {100};
            const int channels[] = {0};
            float lbp0Range[] = {0,1};
            const float *ranges[] = { lbp0Range};
            calcHist(&img_lbp, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            _hist = _hist/cv::norm(_hist,NORM_L2);

            break;

          }



    case HSVLBP:
        {
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(_img, image0, COLOR_BGR2HSV);
            int imgCount = 4;
            int dims = 4;
            const int sizes[] = {30,30,40,80};
            const int channels[] = {0,1,2,3};
            float hRange[] = {0,1};
            float sRange[] = {0,1};
            float vRange[] = {0,1};
            float lbpRange0[] = {0,1};
            float lbpRange1[] = {0,1};
            const float *ranges[] = {hRange,sRange,lbpRange0, lbpRange1};


            split(image0, splitHSV);
            splitHSV[0].convertTo(splitHSV[0], CV_32F, 1/180);
            splitHSV[1].convertTo(splitHSV[1], CV_32F, 1/256);


            Mat img_gray, img_lbp0, img_lbp1;
            cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            LocalBinaryPattern pattern0 (1,8, true);
            LocalBinaryPattern pattern1 (2,16, true);
            pattern0.UniformLBP(img_gray, img_lbp0);
            pattern1.UniformLBP(img_gray, img_lbp1);
            img_lbp0.convertTo(img_lbp0, CV_32F, 1.0/255);
            img_lbp1.convertTo(img_lbp1, CV_32F, 1.0/255);



            Mat imageSet[] = {splitHSV[0],splitHSV[1], img_lbp0, img_lbp1};

            calcHist(imageSet, imgCount, channels, _mask, _hist, dims, sizes, ranges);
            _hist = _hist/cv::norm(_hist,NORM_L2);

            break;
        }
    default:
        cout << "NOTE: no other features are implemented so far" <<endl;
        break;



    }

}





void MyHistogram::ComputeHist(cv::Mat& image, FeatureSpace method, cv::Mat& mask, cv::Mat& hist)
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
            calcHist(&image, imgCount, channels, mask, hist, dims, sizes, ranges);
            break;
        }

    case HSV:

        {
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(image, image0, COLOR_BGR2HSV);
            int imgCount = 3;
            int dims = 3;
            const int sizes[] = {30,50,50};
            const int channels[] = {0,1,2};
            float hRange[] = {0,1};
            float sRange[] = {0,1};
            float vRange[] = {0,1};
            const float *ranges[] = {hRange,sRange, vRange};


            split(image0, splitHSV);
            splitHSV[0].convertTo(splitHSV[0], CV_32F, 1.0f/180);
            splitHSV[1].convertTo(splitHSV[1], CV_32F, 1.0f/256);
            splitHSV[2].convertTo(splitHSV[2], CV_32F, 1.0f/256);


            Mat imageSet[] = {splitHSV[0],splitHSV[1], splitHSV[2]};

            calcHist(imageSet, imgCount, channels, mask, hist, dims, sizes, ranges);
            break;
        }


    case LBP:
        /// only single layer uniform local binary pattern, R = 1, P = 8
        {


            Mat img_gray, img_lbp;
            cv::cvtColor(image, img_gray, COLOR_BGR2GRAY);
            int radius = 1;
            int points = 8;
            LocalBinaryPattern pattern0(radius, points, false);
            pattern0.UniformLBP(img_gray, img_lbp);
            img_lbp.convertTo(img_lbp, CV_32F, 1.0/255);

            /// histogram parameters
            int imgCount = 1;
            int dims = 1;
            const int bin0 = pattern0.GetNumBins();
            int range0 = pattern0.GetRanges();

            const int sizes[] = {100};
            const int channels[] = {0};
            float lbp0Range[] = {0,1};
            const float *ranges[] = { lbp0Range};
            calcHist(&img_lbp, imgCount, channels, mask, hist, dims, sizes, ranges);
            hist = hist/cv::norm(_hist,NORM_L2);

            break;

          }



    case HSVLBP:
        {
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(image, image0, COLOR_BGR2HSV);
            int imgCount = 4;
            int dims = 4;
            const int sizes[] = {30,30,40,80};
            const int channels[] = {0,1,2,3};
            float hRange[] = {0,1};
            float sRange[] = {0,1};
            float vRange[] = {0,1};
            float lbpRange0[] = {0,1};
            float lbpRange1[] = {0,1};
            const float *ranges[] = {hRange,sRange,lbpRange0, lbpRange1};


            split(image0, splitHSV);
            splitHSV[0].convertTo(splitHSV[0], CV_32F, 1/180);
            splitHSV[1].convertTo(splitHSV[1], CV_32F, 1/256);


            Mat img_gray, img_lbp0, img_lbp1;
            cvtColor(image, img_gray, COLOR_BGR2GRAY);
            LocalBinaryPattern pattern0 (1,8, true);
            LocalBinaryPattern pattern1 (2,16, true);
            pattern0.UniformLBP(img_gray, img_lbp0);
            pattern1.UniformLBP(img_gray, img_lbp1);
            img_lbp0.convertTo(img_lbp0, CV_32F, 1.0/255);
            img_lbp1.convertTo(img_lbp1, CV_32F, 1.0/255);



            Mat imageSet[] = {splitHSV[0],splitHSV[1], img_lbp0, img_lbp1};

            calcHist(imageSet, imgCount, channels, mask, hist, dims, sizes, ranges);
            hist = hist/cv::norm(_hist,NORM_L2);

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





void MyHistogram::ComputeHistWeights(const cv::Mat& weights)
/// assume that weights are in (0,1) with float
{
    int n_level = 5;
    float h = 1.0f/n_level;
    Mat w, img_temp,hist_temp;
    vector<Mat> histograms;

    for(int i =0; i < n_level; i++)
    {

        threshold(weights, w, i*h, 1.0, THRESH_TOZERO);
        w.convertTo(w,CV_8UC1, 255);

        ComputeHist(_img, _method, w,hist_temp);

        histograms.push_back(hist_temp.clone());
    }

    _hist = hist_temp.clone();

    for(int i = 0; i<histograms.size()-1; i++){

        _hist =_hist + histograms[i];

    }

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
            const float *ranges[] = {rRange,gRange};

            cv::calcBackProject(&in,imgCount,channels,_hist,backproj,ranges);
            backproj.convertTo(backproj, CV_32F, 1.0/255);
            break;
        }

    case HSV:

        {
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(in, image0, COLOR_BGR2HSV);
            int imgCount = 3;
            int dims = 3;
            const int sizes[] = {90,128,1};
            const int channels[] = {0,1,2};
            float hRange[] = {0,180};
            float sRange[] = {0,256};
            float vRange[] = {0,256};
            const float *ranges[] = {hRange,sRange,vRange};


            split(image0, splitHSV);
            Mat imageSet[] = {splitHSV[0],splitHSV[1], splitHSV[2]};

            cv::calcBackProject(imageSet,imgCount,channels,_hist,backproj,ranges);
            backproj.convertTo(backproj, CV_32F, 1.0/255);
            break;



        }


    case LBP:
        /// only single layer uniform local binary pattern, R = 1, P = 8
        /// single pixel backprojection does not work here, we use patch-based similarity
        {

            /// encodes to local binary pattern
            Mat img_gray, img_lbp, back_sparse;
            cv::cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            int radius = 1;
            int points = 8;
            LocalBinaryPattern pattern0(radius, points, false);
            pattern0.UniformLBP(img_gray, img_lbp);
            img_lbp.convertTo(img_lbp, CV_32F, 1.0/255);
            back_sparse = cv::Mat::zeros(img_gray.size(), CV_32F);
            /// histogram parameters
            int imgCount = 1;
            int dims = 1;
            const int bin0 = pattern0.GetNumBins();
            double range0 = pattern0.GetRanges();

            const int sizes[] = {100};
            const int channels[] = {0};
            float lbp0Range[] = {0,1};
            const float *ranges[] = { lbp0Range};

            /// define search space
            cv::Rect bd0;
            SingleBoundingBoxFromROI(_mask, bd0);
            Point c0 = (bd0.br()+bd0.tl())/2.0;
            const int hx = 3;
            const int hy = 3;

            std::vector<cv::Point> pivots;
            /// search the nearest 100 patches
            cv::Mat temp = Mat::zeros(10,10, CV_32F);

            int x0 = max( min(bd0.tl().x-5*hx, nx-bd0.width), bd0.width); // the new x should be within the image domain
            int y0 = max( min(bd0.tl().y-5*hy, ny-bd0.height), bd0.height); // the new y should be within the image domain


            for(int i = -20; i < 20; i++)
            {
                for(int j = -20; j < 20; j++)
                {
                    int x = max( min(c0.x+i*hx, nx-bd0.width), bd0.width); // the new x should be within the image domain
                    int y = max( min(c0.y+j*hy, ny-bd0.height), bd0.height); // the new y should be within the image domain



                    Point c (x,y);
                    pivots.push_back(c);
                    Rect bd = bd0 + (c-c0);
                    Mat patch = img_lbp(bd);
                    Mat hist;

                    calcHist(&patch, imgCount, channels, cv::Mat(), hist, dims, sizes, ranges);
                    hist = hist/cv::norm(hist,NORM_L2);
                    float dist = (float) compareHist(hist,_hist, HISTCMP_BHATTACHARYYA );
                    backproj.at<float>(c) = 1.0f/dist;
//                    cout << backproj.at<float>(c) <<endl;
//                    temp.at<float>(j+5, i+5) = sqrtf(1-dist*dist);

                }
            }


            /// image interpolation

//            for(int i = 0; i < nx; i++){
//                for(int j = 0; j < ny; j++){
//                    float x = (float)(i-x0)/(hx);
//                    float y = (float)(j-y0)/(hy);
//                    backproj.at<float>(j,i) = temp.at<float>(Point2f(x,y));
//                }
//            }
//
//            imshow("back", backproj);
//            imshow("backs", back_sparse);
//            imshow("temp", temp);
//            waitKey(0);
            normalize(backproj, backproj, 0,1,NORM_MINMAX);
            break;
        }



    case HSVLBP:
        {
            Mat image0;
            vector<Mat> splitHSV;
            cv::cvtColor(_img, image0, COLOR_BGR2HSV);
            int imgCount = 4;
            int dims = 4;
            const int sizes[] = {30,30,40,80};
            const int channels[] = {0,1,2,3};
            float hRange[] = {0,1};
            float sRange[] = {0,1};
            float vRange[] = {0,1};
            float lbpRange0[] = {0,1};
            float lbpRange1[] = {0,1};
            const float *ranges[] = {hRange,sRange,lbpRange0, lbpRange1};


            split(image0, splitHSV);
            splitHSV[0].convertTo(splitHSV[0], CV_32F, 1/180);
            splitHSV[1].convertTo(splitHSV[1], CV_32F, 1/256);


            Mat img_gray, img_lbp0, img_lbp1;
            cvtColor(_img, img_gray, COLOR_BGR2GRAY);
            LocalBinaryPattern pattern0 (1,8, true);
            LocalBinaryPattern pattern1 (2,16, true);
            pattern0.UniformLBP(img_gray, img_lbp0);
            pattern1.UniformLBP(img_gray, img_lbp1);
            img_lbp0.convertTo(img_lbp0, CV_32F, 1.0/255);
            img_lbp1.convertTo(img_lbp1, CV_32F, 1.0/255);



            Mat imageSet[] = {splitHSV[0],splitHSV[1], img_lbp0, img_lbp1};

             /// define search space
            cv::Rect bd0;
            SingleBoundingBoxFromROI(_mask, bd0);
            Point c0 = (bd0.br()+bd0.tl())/2.0;
            const int hx = 5;
            const int hy = 5;

            std::vector<cv::Point> pivots;
            /// search the nearest 100 patches
            cv::Mat temp = Mat::zeros(10,10, CV_32F);

            int x0 = max( min(bd0.tl().x-5*hx, nx-bd0.width), bd0.width); // the new x should be within the image domain
            int y0 = max( min(bd0.tl().y-5*hy, ny-bd0.height), bd0.height); // the new y should be within the image domain


            for(int i = -5; i < 5; i++)
            {
                for(int j = -5; j < 5; j++)
                {
                    int x = max( min(c0.x+i*hx, nx-bd0.width), bd0.width); // the new x should be within the image domain
                    int y = max( min(c0.y+j*hy, ny-bd0.height), bd0.height); // the new y should be within the image domain



                    Point c (x,y);
                    pivots.push_back(c);
                    Rect bd = bd0 + (c-c0);
                    Mat patch_h = splitHSV[0](bd);
                    Mat patch_s = splitHSV[1](bd);
                    Mat patch_0 = img_lbp0(bd);
                    Mat patch_1 = img_lbp1(bd);
                    Mat patchSet[] = {patch_h, patch_s, patch_0, patch_1};

                    Mat hist;

                    calcHist(patchSet, imgCount, channels, cv::Mat(), hist, dims, sizes, ranges);
                    hist = hist/cv::norm(hist,NORM_L2);
                    float dist = (float) compareHist(hist,_hist, HISTCMP_BHATTACHARYYA );
                    backproj.at<float>(c) = 1.0f/dist;
//                    cout << backproj.at<float>(c) <<endl;
//                    temp.at<float>(j+5, i+5) = sqrtf(1-dist*dist);

                }
            }
            normalize(backproj, backproj, 0,1,NORM_MINMAX);
            break;


        }




    default:
        cout << "NOTE: no other features are implemented so far" <<endl;
        break;

    }


}





