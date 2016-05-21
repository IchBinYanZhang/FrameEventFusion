#ifndef FRAMESTREAM_H
#define FRAMESTREAM_H

#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include <iostream>
#include <vector>
#include <fstream>

class FrameStream
{
    friend class FrameEventFuser;
    friend class StereoVision;
    public:
        FrameStream();
        FrameStream(std::string& filename_video, std::string& filename_timestamp);
        virtual ~FrameStream();
        void GetTimestamp(std::vector<uint64_t>& out);
        void GetFrameFile(std::string& out);
        void FrameShow();
    protected:
    private:
        cv::VideoCapture cap;
        std::vector<uint64_t> time_stamp;
        std::string _filename;
};

#endif // FRAMESTREAM_H
