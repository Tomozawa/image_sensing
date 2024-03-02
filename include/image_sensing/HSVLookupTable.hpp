#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>

#include <array>

namespace hue_lut{
    template<typename Tp, size_t R, size_t C>
    using LUT_TYPE_ = cv::Matx<Tp, R, C>;

    template<typename Tp>
    using LUT_VALUE_PAIR_ = std::array<Tp, 2>;

    template<typename Tp, size_t R>
    using LUT_PARAM_ = std::array<LUT_VALUE_PAIR_<Tp>, R>;

    template<typename Tp, size_t R, size_t C>
    inline void calc_lut(const LUT_PARAM_<Tp, R>& param, LUT_TYPE_<Tp, R, C>& lut){
        LUT_TYPE_<Tp, R, C> input{};

        for(size_t r = 0; r < R; r++){
            const Tp min = param.at(r).at(0);
            const Tp max = param.at(r).at(1);

            for(size_t c = 0; c < C; c++){
                const unsigned access_index = (min + c) % C;
                if(access_index == max) break;
                input(r, c) = UINT8_MAX;
            }
        }

        lut = input;
    }

    template<typename Tp, size_t R, size_t C>
    inline LUT_TYPE_<Tp, R, C> calc_lut(const LUT_PARAM_<Tp, R>& param){
        LUT_TYPE_<Tp, R, C> result{};
        calc_lut<Tp, R, C>(param, result);
        return result;
    }
} //hue_lut