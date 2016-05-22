#include "FrameEventFuser.h"





bool IsNonZero (int i) { return (i!=0); }


void Erosion( cv::Mat& src, cv::Mat& dst, int kernel_size )
{

  cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*kernel_size + 1, 2*kernel_size+1 ),
                                       cv::Point( kernel_size, kernel_size ) );
  /// Apply the erosion operation
  cv::erode( src, dst, element );

}

void Dilation( cv::Mat& src, cv::Mat& dst, int kernel_size )
{

    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*kernel_size + 1, 2*kernel_size+1 ),
                                       cv::Point( kernel_size, kernel_size ) );
  /// Apply the dilation operation
    cv::dilate( src, dst, element );

}




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



void FrameEventFuser::Matching(cv::Mat& f_pre, cv::Mat& f_cur, cv::Mat& events, cv::Mat& edges)
{
    /// temporal difference computation
    cv::Mat f1, f2;
    f_pre.convertTo(f1,CV_32F);
    f_cur.convertTo(f2,CV_32F);
    cv::Mat ft = f2-f1+128;

    /// thresholding the image
    float* pt;
    for(int i = 0; i < ft.rows; i++)
    {
        pt = ft.ptr<float>(i);

        for(int j = 0; j < ft.cols; j++)
        {
            if (std::abs(pt[j]-128.0f) < 20.0f)
            {
                pt[j] = 128;
            }
            else
            {
                if(pt[j]-128.0f>20.0f)
                    pt[j] = 255;
                else
                    pt[j] = 0;
            }

        }

    }

    /// interpolation of events image
    cv::Mat events_itp;
    cv::resize(events, events_itp, f_cur.size() );
    ft.convertTo(ft,CV_8U);
    cv::imshow("ft", ft);
    cv::imshow("events",events_itp);
    cv::waitKey(50);


}




void FrameEventFuser::FrameEventShow(bool show_ft)
{
    cv::Mat frame;
    cv::Mat event (128,128, CV_8UC1, cv::Scalar(128));
    cv::Mat frame_pre;
    cv::Mat ft;
    cv::VideoCapture cap (this->_filename_frame);
    namedWindow("Frame Visualization", cv::WINDOW_NORMAL);
    namedWindow("Temporal Difference Visualization", cv::WINDOW_NORMAL);
    namedWindow("Event Visualization", cv::WINDOW_NORMAL);
    int nt = this->_event_x.size();
    int num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    int num_triggers1 = count_if (this->_event_trigger.begin(), this->_event_trigger.end(), IsNonZero);
    int N = 1;
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


    /// check #triggers in DVS and frames in video. If they are not equal, the first triggered event is discarded.
    std:: cout << num_frames << "   " << num_triggers1 << std::endl;
    if (num_triggers1 != num_frames)
    {


        while (i < nt)
        {
            if (this->_event_trigger[i]==0)
            {
                j++;
            }
            else
            {
                j=0;
                break;
            }
            i++;
        }
    }
    i++;


    /// main loop
    int ff = 0;
    while(i < nt)
    {
        if (this->_event_trigger[i]==0)
        {
            j++;
        }
        else
        {
            std::cout << "-----------------------------------"<<std::endl;
            cap >> frame;
            if (!frame.empty())
            {
                cv::imshow("Frame Visualization", frame);ff++;


                cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
//                    this->Matching(frame_pre, frame, event, edges);
//                    edges.convertTo(edges_draw,CV_8U);
                frame.convertTo(frame,CV_32F);
                if (show_ft && ff>1)
                {
                    ft = frame-frame_pre+128;
                    ft.convertTo(ft, CV_8U);
                    cv::imshow("Temporal Difference Visualization",ft);


                }

            }
            else
                std::cout << " - Current Frame is empty." << std::endl;

            int event_per_frame = floor(j/N);
            for (int f = 0; f < N; f++)
            {
                for (int p = ii+f*event_per_frame; p < ii+(f+1)*event_per_frame; p++  )
                    event.at<uint8_t>(127-this->_event_y[p],127-this->_event_x[p]) += uint8_t(255*(float(this->_event_pol[p])-0.5f));


                cv::imshow("Event Visualization",event);
                cv::waitKey(10);
            }
            Matching(frame_pre, frame, event, event);

            cv::waitKey(10);


            std::cout << " - frame index = " << uint16_t(ff) << std::endl;
            std::cout << " - #atomic events = " << uint16_t(j) <<std::endl;
            std::cout << " - current event index = " << (i) <<std::endl;
            j = 0;
            num_triggers++;
            ii = i;
            frame_pre = frame;
            event.setTo(cv::Scalar(128));

        }
        i++;
    }
    std::cout << "-----------------------------------"<<std::endl;
    std::cout << " - " << ff <<" frames are displayed." << std::endl;
    std::cout << " - " << num_triggers << " are found." <<std::endl;

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
