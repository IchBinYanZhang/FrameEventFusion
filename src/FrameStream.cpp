#include "FrameStream.h"

FrameStream::FrameStream()
{
    this->cap.release();
    this->time_stamp.clear();
}



FrameStream::FrameStream(std::string& file_video, std::string& file_timestamp)
{
    this->time_stamp.clear();
    cap.open(file_video);

    if (!cap.isOpened())
    {
        std::cerr << "Video file is not opened!" <<std::endl;
        exit(-1);
    }

    std::ifstream file;
    file.open(file_timestamp);
    if (!file.is_open())
    {
        std::cerr << "TimeStamp file is not opened!" <<std::endl;
        exit(-2);
    }

    std::string ts;

    while(!file.eof())
    {
        file>>ts;
        int temp432, temp754;
        if (ts.size () ) {
            sscanf (ts.c_str(), "%i,%i", &temp432, &temp754);
            this->time_stamp.push_back(temp432);
        }
    }

    this->_filename = file_video;
    this->_filename_ts = file_timestamp;

}


void FrameStream::FrameShow()
{
    cv:: Mat frame;
    namedWindow("Frame Visualization", cv::WINDOW_NORMAL);
    int num_frames = this->time_stamp.size();
    std:: cout << num_frames << std::endl;
    for(int i = 0; i < num_frames; i++)
    {
        this->cap >> frame;
        frame.convertTo(frame,CV_8UC3);
        if (!frame.empty())
        {
//            std::cout << "- frame index = "<<this->time_stamp[i]- this->time_stamp[1]<< std::endl;
            cv::imshow("Frame Visualization", frame);
            cv::waitKey(10);
        }
    }
//    std::cout << "- Display ends." <<std::endl;
//    std::cout << "- #frames = " <<num_frames << std::endl;
//
}


void FrameStream::GetTimestamp(std::vector<uint64_t>& out)
{
    out = this->time_stamp;
}

void FrameStream::GetFrameFile(std::string& out)
{
    out = this->_filename;
}

void FrameStream::GetTimestampFile(std::string& out)
{
    out = this->_filename_ts;
}






FrameStream::~FrameStream()
{
    this->cap.release();
    this->time_stamp.clear();
}



