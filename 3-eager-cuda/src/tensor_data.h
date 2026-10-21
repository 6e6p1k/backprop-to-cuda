#pragma once

#include <vector>
#include <string>
#include <stdexcept>

enum class Device { CPU, CUDA };

inline Device deviceFromString(const std::string& s) {
    if (s == "cpu") return Device::CPU;
    if (s == "cuda") return Device::CUDA;
    throw std::invalid_argument("Unknown device: " + s);
}

inline std::string deviceToString(Device device) {
    if (device == Device::CPU) return "cpu";
    if (device == Device::CUDA) return "cuda";
    throw std::invalid_argument("Unknown device");
}

struct TensorData {
    std::vector<float> values;
    std::vector<int> strides;
    float* cuda_data = nullptr;
    int* cuda_strides = nullptr;
    Device device = Device::CPU;

    std::vector<int> shape;
    std::vector<float> grad;

    int numel() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    ~TensorData();  // defined in cuda_ops.cu
};
