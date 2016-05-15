#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <opencv/cv.h>

#define NUM_BYTES_PER_EVENT 8    //AEDAT 2.0

using namespace std;
int main(int argc, char** argv)
{

    /// argument transfer
    std::string filename= argv[1];
    int maxevents = atoi(argv[2]);
    std::cout << "Read out at most " << maxevents <<" events from file "<<filename <<std::endl;

    /// read the file
    std::ifstream file(filename.c_str(),  std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << filename << " is not opened" <<std::endl;
        return -1;
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
    int num_events = floor((file.tellg()-header_end_pos)/NUM_BYTES_PER_EVENT);
    num_events = min(num_events, maxevents);
    std::vector<unsigned char> x;
    std::vector<unsigned char> y;
    std::vector<unsigned char> pol;
    std::vector<unsigned char> trigger;
    char memblock;

    file.seekg(header_end_pos+3);
    uint64_t pos = file.tellg();


    for (int i = 1; i <= num_events; i++)
    {




        /// read y and trigger
        file.seekg(pos);
        file.read(&memblock,1);
        y.push_back(memblock & 0x7f);
        trigger.push_back( (memblock >> 7) & 0x01 );



        /// read x and pol
        pos++;  /// file pointer shifted by one Char (8 bits)
        file.read(&memblock,1);
        pol.push_back((memblock & 0x01));
        x.push_back(((memblock >> 1) & 0x7f));


        /// move the file pointer to the next CORRECT position
        pos += 7;
        file.seekg(pos); // skip the timestamp memory

    }


    for (int t = 0; t < 100; t++)
        std::cout<<"time stamp=" <<t << "\t" << "x=" << int(x[t]) << "\t"<< "y=" <<int(y[t]) << "\t" <<"pol=" << int(pol[t]) << "\t"<<"trigger="<< int(trigger[t]) << std::endl;




    file.close();

    return 0;


}
