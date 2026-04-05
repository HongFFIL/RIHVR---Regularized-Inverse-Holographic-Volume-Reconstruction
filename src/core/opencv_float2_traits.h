// opencv_float2_traits.h
#pragma once

#include <opencv2/core.hpp>
#include <cuda_runtime.h>  // for float2

namespace cv {

template<>
class DataType<float2>
{
public:
    typedef float2 value_type;
    typedef value_type work_type;
    typedef float channel_type;

    enum {
        generic = 0,
        depth = CV_32F,
        channels = 2,
        fmt = (int)'f'
    };

    typedef Vec<float, channels> vec_type;
};

}  // namespace cv