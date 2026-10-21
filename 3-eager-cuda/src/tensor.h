#pragma once

// imports
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

// namespaces
#include "./tensor_data.h"
#include "./meta.h"
#include "./ops.h"
#include "cuda_ops.cuh"

namespace autograd { struct Node; }

class Tensor {
public:
    std::shared_ptr<TensorData> td;
    std::shared_ptr<autograd::Node> grad_fn;
    bool requires_grad = false;

    // basic init
    Tensor(std::vector<float> values, std::vector<int> shape, std::string device = "cpu", bool requires_grad = false);

    // init filled with a number
    Tensor(std::vector<int> shape, float fill_val, std::string device = "cpu", bool requires_grad = false);

    int numel() const;

    Tensor to_cuda() const;

    Tensor to_cpu() const;

    std::vector<float> to_host_truncated(int max_elements) const;

    std::vector<float> to_host_full() const;

    float at(std::vector<int> idxs) const;

    void print_values() const;

    void print_shape() const;

    Tensor add(const Tensor& other) const;

    Tensor mul(const Tensor& other) const;

    Tensor sum(int dim, bool keepdims) const;

    Tensor contiguous() const;

    Tensor transpose() const;

    Tensor mean(int dim, bool keepdims) const;

    Tensor max(int dim, bool keepdims) const;

    Tensor argmax(int dim) const;

    Tensor eq(const Tensor& other) const;

    Tensor log() const;

    Tensor reshape(std::vector<int> new_shape) const;

    Tensor squeeze(int dim) const;

    Tensor unsqueeze(int dim) const;

    Tensor flatten(int start_dim, int end_dim) const;

    Tensor matmul(const Tensor& other) const;

    Tensor matmul_fast(const Tensor& other) const;

    Tensor matmul_blas(const Tensor& other) const;

    Tensor matmul_cuda(const Tensor& other) const;

    Tensor exp() const;

    Tensor pow(const float num) const;

    void reset_grad();

    void sgd_update(float lr);

    void backward();

private:
    Tensor(std::vector<float> values, std::vector<int> shape, std::vector<int> strides);

    Tensor(float* cuda_ptr, std::vector<int> shape);

    Tensor(float* cuda_ptr, std::vector<int> shape, int* strides);

    Tensor dispatch_matmul(const Tensor& other) const;

    template <typename Op>
    Tensor dispatch_elementwise(const Tensor& other) const {
        auto [res_shape, ndim] = meta::broadcast_shape(td->shape, other.td->shape);
        int res_size = 1; for (int s : res_shape) res_size *= s;
        if (td->device == Device::CPU) {
            auto [result_data, _] = ops::_elementwise(td->values, td->shape, other.td->values, other.td->shape, Op{});
            return Tensor(result_data, res_shape);
        } else if (td->device == Device::CUDA) {
            float* ptr = cuda_ops::elemwise_launch<Op>(
                td->cuda_data, other.td->cuda_data, res_size, ndim,
                td->shape, other.td->shape, res_shape,
                td->cuda_strides, other.td->cuda_strides);
            return Tensor(ptr, res_shape);
        }
        throw std::logic_error("Unknown device.");
    }

    Tensor dispatch_sum(int dim, bool keepdims) const;
};
