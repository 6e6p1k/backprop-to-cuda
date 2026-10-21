// include only once
#pragma once

#include <vector>

// shared poniters
struct TensorData {
    std::vector<float> values;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<float> grad;
};