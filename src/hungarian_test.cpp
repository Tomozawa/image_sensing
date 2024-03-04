#include <Hungarian.hpp>
#include <iostream>

int main(){
    cv::Mat cost_matrix = (cv::Mat_<int16_t>(2, 3) << 3, 2, 1, 6, 5, 4);

    const auto result = hungarian::hungarian::assign(cost_matrix, false);

    for(const auto& point : result){
        std::cout << cv::format(cv::Mat(point), cv::Formatter::FMT_DEFAULT) << std::endl;
    }
}