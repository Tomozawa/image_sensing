#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>

#include <array>

namespace hue_lut{
    using LUT_TYPE_ = cv::Mat;

    using LUT_VALUE_PAIR_ = std::pair<uint8_t, uint8_t>;

    template<size_t Cn>
    using LUT_PARAM_ = std::array<LUT_VALUE_PAIR_, Cn>;

    template<size_t Cn>
    inline void calc_lut(const LUT_PARAM_<Cn>& param, LUT_TYPE_& lut){
        static_assert(Cn <= 4U);
        LUT_TYPE_ input(cv::Size(1, 256), CV_8UC(Cn));

        std::array<bool, Cn> end_flags = {false,};
        for(int i = 0; i < lut.cols; i++){
            for(size_t c = 0; c < Cn; c++){
                if(end_flags.at(c)) continue;
                const size_t access_index = (i + param.at(c).first) % 256U;
                input.at<uint8_t>(0, 3U * access_index + c) = UINT8_MAX;

                if(access_index == param.at(c).second) end_flags.at(c) = true;
            }
        }

        lut = input;
    }

    template<size_t Cn>
    inline LUT_TYPE_ calc_lut(const LUT_PARAM_<Cn>& param){
        LUT_TYPE_ result{};
        calc_lut(param, result);
        return result;
    }
} //hue_lut