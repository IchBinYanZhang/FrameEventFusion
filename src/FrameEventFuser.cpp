#include "FrameEventFuser.h"

FrameEventFuser::FrameEventFuser()
{
    //ctor
}


FrameEventFuser::FrameEventFuser(FrameStream& frames, EventStream& events)
{
    frames.GetFrameFile(this->_filename_frame);
    frames.GetTimestamp(this->_time_stamp);
    events.GetX(this->_event_x);
    events.GetY(this->_event_y);
    events.GetPolarity(this->_event_pol);
    events.GetTrigger(this->_event_trigger);

}

void FrameEventFuser::FrameEventShow()
{
    cv::Mat frame;
    cv::Mat event (128,128, CV_8UC1, cv::Scalar(128));
    cv::VideoCapture cap (this->_filename_frame);
    namedWindow("Frame Visualization", cv::WINDOW_NORMAL);
    namedWindow("Event Visualization", cv::WINDOW_NORMAL);
    int nt = this->_event_x.size();
    int num_frames = this->_time_stamp.size();
    int N = 3;
    int i = 0;
    int ii = 0;
    int j = 0;
    int num_triggers = 0;
//    for(int i = 0; i < num_frames; i++)
//    {
//        this->cap >> frame;
//        frame.convertTo(frame,CV_8UC3);
//
//        cv::imshow("Frame Visualization", frame);
//        cv::waitKey(10);
//
//    }
//

    int ff = 0;
    while(i < nt)
    {
        if (this->_event_trigger[i]==0)
        {
            j++;
        }
        else
        {
            cap >> frame; ff++;
            cv::imshow("Frame Visualization", frame);
            int event_per_frame = floor(j/N);
            for (int f = 0; f < N; f++)
            {
                for (int p = ii+f*event_per_frame; p < ii+(f+1)*event_per_frame; p++  )
                    event.at<uint8_t>(127-this->_event_y[p],127-this->_event_x[p]) += uint8_t(255*(float(this->_event_pol[p])-0.5f));

                cv::imshow("Event Visualization",event);
                cv::waitKey(3);
                event.setTo(cv::Scalar(128));
            }

            cv::waitKey(10);

            std::cout << "-----------------------------------"<<std::endl;
            std::cout << " - frame index = " << uint16_t(ff) << std::endl;
            std::cout << " - #atomic events = " << uint16_t(j) <<std::endl;
            std::cout << " - current event index = " << (i) <<std::endl;
            j = 0;
            num_triggers++;
            ii = i;
        }
        i++;
    }
    std::cout << "-----------------------------------"<<std::endl;
    std::cout << " - Display ends." << std::endl;


//
}



FrameEventFuser::~FrameEventFuser()
{

    this->_time_stamp.clear();
    this->_event_x.clear();
    this->_event_y.clear();
    this->_event_pol.clear();
    this->_event_trigger.clear();
}
