#include <Hungarian.hpp>
#include <iostream>

int main(){
    cv::Mat cost_matrix{cv::Size(0, 0), CV_16S};

    const auto result = hungarian::hungarian::assign(cost_matrix, false);

    for(const auto& point : result){
        std::cout << cv::format(cv::Mat(point), cv::Formatter::FMT_DEFAULT) << std::endl;
    }
}