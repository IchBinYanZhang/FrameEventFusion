#ifndef EVENTSTREAM_H
#define EVENTSTREAM_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
class EventStream
{
    friend class FrameEventFuser;
    public:
        EventStream();
        virtual ~EventStream();
        void ReadFromFile(std::string& filename, int max_event);
        void Clear();
        void EventShow(int acc_time);
        void EventShow();
        void GetX(std::vector<uint8_t>& out);
        void GetY(std::vector<uint8_t>& out);
        void GetPolarity(std::vector<uint8_t>& out);
        void GetTrigger(std::vector<uint8_t>& out);

    protected:
    private:
    std::vector<uint8_t> x;
    std::vector<uint8_t> y;
    std::vector<uint8_t> pol;
    std::vector<uint8_t> trigger;
    int nx;
    int ny;

};

#endif // EVENTSTREAM_H
