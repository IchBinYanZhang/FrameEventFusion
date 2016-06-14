#include "LocalBinaryPattern.h"

LocalBinaryPattern::LocalBinaryPattern(int radius, int n_points, bool uniform)
{
    _r = radius;
    _n_pts = n_points;
    _uniform = uniform;
}



LocalBinaryPattern::~LocalBinaryPattern( )
{

}


// Brian Kernighan's method to count set bits
int LocalBinaryPattern::countSetBits(int code)
{
  int count=0;
  int v=code;
  for(count=0;v;count++)
  {
  v&=v-1; //clears the LSB
  }
  return count;
}

inline int LocalBinaryPattern::rightshift(int num, int shift)
{
    return (num >> shift) | ((num << (8 - shift)&0xFF));
}


bool LocalBinaryPattern::checkUniform(int code)
{
    int b = rightshift(code,1);
  ///int d = code << 1;
  int c = code ^ b;
  //d= code ^d;
  int count=countSetBits(c);
  //int count1=countSetBits(d);
  if (count <=2 )
      return true;
  else
      return false;
}


//void LocalBinaryPattern::initUniform(int N)
///// N = the number of points in one circle
//{
//
//    _lookup = cv::Mat::zeros(N,1);
//    int range_bits = pow(2,N);
//    int index=0;
//    for(int i=0;i<=range_bits;i++)
//    {
//        bool status=checkUniform(i);
//        if(status==true)
//        {
//            lookup[i]=index;
//            index++;
//        }
//        else
//        {
//            lookup[i]=59;
//        }
//    }
//
//    initHistogram();
//
//}


int LocalBinaryPattern::GetNumBins()
{
    if(_uniform)
        return _n_pts*_n_pts - _n_pts + 2;
    else
        return  (int)pow(2,_n_pts);
}

double LocalBinaryPattern::GetRanges()
{
    return pow(2,_n_pts);
}

void LocalBinaryPattern::UniformLBP( Mat& src, Mat& dst) {



    int radius = _r;

    int neighbors = max(min(_n_pts,31),1); // set bounds...
    src.convertTo(src, CV_32F);
    dst = Mat::zeros(src.rows, src.cols, CV_32SC1);
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
                dst.at<unsigned int>(i,j) += ((t > src.at<float>(i,j)) && (abs(t-src.at<float>(i,j)) > std::numeric_limits<float>::epsilon())) << n;


                if(_uniform){
                    if(!checkUniform(dst.at<unsigned int>(i,j)))
                        dst.at<unsigned int>(i,j) = 5;
                }
            }
        }
    }
}



//
//
//
//void LocalBinaryPattern::initHistogram()
//{
//    vector<int> channel;
//    channel.push_back(0);
//    hist.setChannel(channel);
//    vector<int> size;
//    size.push_back(sizeb+1);
//    hist.setHistSize(size);
//
//}
//
//Mat LocalBinaryPattern::computeHistogram(Mat cell)
//{
//    Mat tmp_hist;
//    hist.BuildHistogram(cell);
//    hist.getHist().copyTo(tmp_hist);
//    //tmp_hist=tmp_hist/tmp_hist.total();
//    return tmp_hist;
//}
//
//
//
//vector<float> LocalBinaryPattern::spatialHistogram(Mat lbpImage,Size grid)
//{
//    vector<float> histogram;
//    histogram.resize(grid.width*grid.height*sizeb);
//    int width=lbpImage.cols/grid.width;
//    int height=lbpImage.rows/grid.height;
//    int cnt=0;
//    //#pragma omp parallel for
//    for(int i=0;i<grid.height;i++)
//    {
//        for(int j=0;j<grid.width;j++)
//        {
//            Mat cell=lbpImage(Rect(j*width,i*height,width,height));
//            Mat cell_hist=computeHistogram(cell);
//            //imshow("FFF",cell_hist);
//            Mat tmp_feature;
//            cell_hist.reshape(1,1).convertTo(tmp_feature,CV_32FC1);
//            float * ptr=tmp_feature.ptr<float>(0);
//            for(int k=0;k<tmp_feature.cols-1;k++)
//            {
//                if(ptr[k]==0)
//                    ptr[k]=1.0/sizeb;
//                histogram[(cnt*sizeb)+k]=ptr[k];
//              //  cerr << ptr[k] << endl;
//            }
//            cnt++;
//        }
//    }
//
//    return histogram;
//}
//
//
//void LocalBinaryPattern::computeBlock(Mat image,Mat & dst,int block)
//{
//    ix.compute(image);
//    image.copyTo(dst);
//    dst.setTo(cv::Scalar::all(0));
//    int width=image.cols;
//    int height=image.rows;
//    for(int i=block;i<height-block;i=i+block)
//    {
//        for(int j=block;j<width-block;j=j+block)
//        {
//            int x=i;
//            int y=j;
//            Rect r=Rect(j,i,block,block);
//            int meanv=ix.calcMean(r);
//            int code=0;
//            for(int k=0;k<8;k++)
//            {
//                switch(k)
//                {
//                case 0:
//                    y=i-block;
//                    x=j-block;
//                break;
//                case 1:
//                    y=i-block;
//                    x=j;
//                break;
//                case 2:
//                    y=i-block;
//                    x=j+block;
//                break;
//                case 3:
//                    y=i;
//                    x=j+block;
//                break;
//                case 4:
//                    y=i+block;
//                    x=j+block;
//                break;
//                case 5:
//                    y=i+block;
//                    x=j;
//                break;
//                case 6:
//                    y=i+block;
//                    x=j-block;
//                break;
//                case 7:
//                    y=i;
//                    x=j-block;
//                break;
//                default:
//                break;
//                }
//                Rect r1=Rect(x,y,block,block);
//                int val=(int)ix.calcMean(r1);
//                code|=(meanv >= val)<<(7-k);
//
//
//            }
//            code=lookup[code];
//            Mat roi=dst(r);
//            roi.setTo(cv::Scalar::all(code));
//
//
//        }
//    }
//
//
//}



