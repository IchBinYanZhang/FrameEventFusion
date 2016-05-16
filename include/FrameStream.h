#ifndef FRAMESTREAM_H
#define FRAMESTREAM_H

#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <vector>
#include <fstream>

class FrameStream
{
    public:
        FrameStream();
        FrameStream(std::string filename_video, std::string filename_timestamp);
        virtual ~FrameStream();
        void GetTimestamp(std::vector<uint64_t> out);
        void GetFrameStream(cv::VideoCapture out);
        void FrameShow();
    protected:
    private:
        cv::VideoCapture cap;
        std::vector<uint64_t> time_stamp;
};

#endif // FRAMESTREAM_H
