#include "StereoVision.h"

StereoVision::StereoVision()
{
    //ctor
}

StereoVision::StereoVision(FrameStream& s1, FrameStream& s2)
{
//    std::string filename1;
//    std::string filename2;
//    std::vector<uint64_t> ts1;
//    std::vector<uint64_t> ts2;
//
//    s1.GetFrameFile(filename1);
//    s1.GetTimestamp(ts1);
//    s2.GetFrameFile(filename2);
//    s2.GetTimestamp(ts2);
//
//    FrameStream s1;
//    FrameS
    this->_stream.push_back(s1);
    this->_stream.push_back(s2);

}

void StereoVision::StereoShow(bool is_rectified)
{
    this->_stream[0].FrameShow();
    this->_stream[1].FrameShow();

}


StereoVision::~StereoVision()
{
    //dtor
}
