#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>

namespace hue_lut{
    typedef cv::Vec<uint8_t, 256> LUT_TYPE;

    template<int cn>
    void calc_hue_lut(int hue_min, int hue_max, cv::Vec<uint8_t, cn>& lut){
        CV_StaticAssert(cn <= UINT8_MAX || cn >= 0, "invalied cn");

        LUT_TYPE input{};

        for(uint8_t i = 0; i < cn; i++){
            const uint8_t access_index = (hue_min + static_cast<unsigned>(i)) % cn;
            if(access_index == hue_max) break;
            input[access_index] = UINT8_MAX;
        }

        lut = input;
    }


} //hue_lut