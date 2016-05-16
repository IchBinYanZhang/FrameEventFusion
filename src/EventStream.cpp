#define NUM_BYTES_PER_EVENT 8    //AEDAT 2.0

#include "EventStream.h"

using namespace std;
using namespace cv;

EventStream::EventStream()
{
    // initialize the vector
    this->x = {0};
    this->y = {0};
    this->pol = {0};
    this->trigger = {0};
    this->nx = 128;
    this->ny = 128;

}

void EventStream::ReadFromFile(std::string filename, int maxevents)
{


    std::cout << "Read out at most " << maxevents <<" events from file "<<filename <<std::endl;

    /// read the file
    std::ifstream file(filename.c_str(),  std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << filename << " is not opened" <<std::endl;
        exit(-1);
    }

    /// read the header
    long streampos;
    long header_end_pos;
    long data_start_pos;
    char line[1024];
    // loop for header reading
    while (file.get(line,1024,'\r\n'))
    {
        if (line[0] == '#')
        {
//            std::cout << line <<std::endl;
            streampos = file.tellg();
            file.seekg(streampos+1);
            header_end_pos = streampos;
            continue;
        }
        else
            break;
    }

    /// reading events and time stamps
    file.seekg(0,std::ios::end);
    int num_events = floor((file.tellg()-header_end_pos)/NUM_BYTES_PER_EVENT); // count the number of events in the file
    num_events = min(num_events, maxevents);
    char memblock;

    // set the file pointer to the correct position
    file.seekg(header_end_pos+3);
    uint64_t pos = file.tellg();

    // main loop of event reading
    for (int i = 1; i <= num_events; i++)
    {

        /// read y and trigger
        file.seekg(pos);
        file.read(&memblock,1);
        this->y.push_back(memblock & 0x7f);
        this->trigger.push_back( (memblock >> 7) & 0x01 );



        /// read x and pol
        pos++;  /// file pointer shifted by one Char (8 bits)
        file.read(&memblock,1);
        this->pol.push_back((memblock & 0x01));
        this->x.push_back(((memblock >> 1) & 0x7f));


        /// move the file pointer to the next CORRECT position
        pos += 7;
        file.seekg(pos); // skip the timestamp memory

    }


//    for (int t = 0; t < 100; t++)
//        std::cout<<"time stamp=" <<t << "\t" << "x=" << int(this->x[t]) << "\t"<< "y=" <<int(this->y[t]) << "\t" <<
//        "pol=" << int(this->pol[t]) << "\t"<<"trigger="<< int(this->trigger[t]) << std::endl;


    file.close();

}

void EventStream::Clear()
{
    this->x.clear();
    this->y.clear();
    this->pol.clear();
    this->trigger.clear();

}


void EventStream::EventShow(int acc_time)
/// acc_time is measured by microsecond and given by an integer.
{
    cv::Mat img (this->nx,this->ny, CV_8UC1, cv::Scalar(128));
    int nt = this->x.size();
    int max_display = std::floor(nt/acc_time);
    cv::namedWindow("Event Visualization", cv::WINDOW_NORMAL);
    for(int i = 0; i < max_display; i++)
    {
        for (int j = 0; j < acc_time; j++)
            img.at<uint8_t>(this->ny-1-this->y[i*acc_time+j],this->x[i*acc_time+j]) += uint8_t(255*(float(this->pol[i*acc_time+j])-0.5f));
        cv::imshow("Event Visualization",img);
        cv::waitKey(0);
        img.setTo(Scalar(128));
    }

//    std::cout << "- Display Ends." <<std::endl;

}



void EventStream::EventShow( )
/// display the events between two triggered signals
/// Between two trigger signals, N frames are displayed.
{
    cv::Mat img (this->nx,this->ny, CV_8UC1, cv::Scalar(128));
    int nt = this->x.size();
    cv::namedWindow("Event Visualization", cv::WINDOW_NORMAL);
    int i = 0;
    int j = 0;
    int ii = 0;
    int num_triggers = 0;
    const int N = 3;
    while(i < nt)
    {
        if (this->trigger[i]==0)
        {
            j++;
        }
        else
        {
            int event_per_frame = floor(j/N);
            for (int f = 0; f < N; f++)
            {
                for (int p = ii+f*event_per_frame; p < ii+(f+1)*event_per_frame; p++  )
                    img.at<uint8_t>(this->ny-1-this->y[p],this->x[p]) += uint8_t(255*(float(this->pol[p])-0.5f));

                cv::imshow("Event Visualization",img);
                cv::waitKey(50);
                img.setTo(Scalar(128));
            }

            cv::waitKey(1000);

            std::cout << "-----------------------------------"<<std::endl;
            std::cout << " - previous trigger index = " << uint16_t(ii) << std::endl;
            std::cout << " - current trigger index = " << uint16_t(i) << std::endl;
            std::cout << " - #atomic events = " << uint16_t(j) <<std::endl;
            j = 0;
            num_triggers++;
            ii = i;
        }
        i++;
    }
    std::cout << "-----------------------------------"<<std::endl;
    std::cout << "- Display Ends." <<std::endl;
    std::cout << "- #triggers = " << num_triggers <<std::endl;


}





void EventStream::GetX(std::vector<uint8_t> out)
{
    out = this->x;
}

void EventStream::GetY(std::vector<uint8_t> out)
{
    out = this->y;
}

void EventStream::GetPolarity(std::vector<uint8_t> out)
{
    out = this->pol;
}

void EventStream::GetTrigger(std::vector<uint8_t> out)
{
    out = this->trigger;
}



EventStream::~EventStream()
{
    this->x.clear();
    this->y.clear();
    this->pol.clear();
    this->trigger.clear();

}
