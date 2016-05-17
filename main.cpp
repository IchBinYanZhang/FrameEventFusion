#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <opencv/cv.h>
#include "EventStream.h"
#include "FrameStream.h"
#include "FrameEventFuser.h"


using namespace std;
int main(int argc, char** argv)
{

    /// argument transfer and setup
    if(argc != 5)
    {
        std::cerr<< "./program [filename_frame] [filename_timestamp] [filename_events] [maximum_events]" <<std::endl;
        return -1;
    }
    std::string filename_frame= argv[1];
    std::string filename_timestamp = argv[2];
    std::string filename_event = argv[3];
    int maxevents = atoi(argv[4]);

    /// event processing
    EventStream stream;
    stream.ReadFromFile(filename_event, maxevents);
//    stream.EventShow();

    /// frame processing
    FrameStream frames(filename_frame, filename_timestamp);
//    frames.FrameShow();

    /// fuser
    FrameEventFuser fuser(frames, stream);
    fuser.FrameEventShow();

    return 0;


}
