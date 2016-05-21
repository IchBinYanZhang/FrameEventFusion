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
        std::cerr<< "./program [filename1] [filename_timestamp1] [filename2] [filename_timestamp2]" <<std::endl;
        return -1;
    }
    std::string filename_frame= argv[1];
    std::string filename_timestamp = argv[2];
    std::string filename_frame2= argv[3];;
    std::string filename_timestamp2 = argv[4];


    FrameStream frames(filename_frame, filename_timestamp);
    FrameStream frames2(filename_frame2, filename_timestamp2);


    return 0;


}
