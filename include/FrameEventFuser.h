#ifndef FRAMEEVENTFUSER_H
#define FRAMEEVENTFUSER_H

#include <cstdlib>
#include <vector>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
#include "FrameStream.h"
#include "EventStream.h"

class FrameEventFuser
{
    public:
        FrameEventFuser();
        FrameEventFuser(FrameStream& frames, EventStream& events);
        void FrameEventShow();
        virtual ~FrameEventFuser();
    protected:
    private:
        std::string _filename_frame;
        std::vector<uint64_t> _time_stamp;
        std::vector<uint8_t> _event_x;
        std::vector<uint8_t> _event_y;
        std::vector<uint8_t> _event_pol;
        std::vector<uint8_t> _event_trigger;

};

#endif // FRAMEEVENTFUSER_H
