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



using namespace std;
int main(int argc, char** argv)
{

    /// argument transfer and setup
//    if(argc != 3)
//    {
//        std::cerr<< "./program [filename] [maximum event]" <<std::endl;
//        return -1;
//    }
    std::string filename= argv[1];
//    int maxevents = atoi(argv[2]);
    std::string filename2 = argv[2];

//    /// event processing
//    EventStream stream;
//    stream.ReadFromFile(filename, maxevents);
//    stream.EventShow();

    /// frame processing
    FrameStream frames(filename, filename2);
    frames.FrameShow();


    return 0;


}
