#pragma once

// libs
#include <vector>
#include <stdexcept>
#include <functional>

// namespaces
#include "./meta.h"


namespace ops {
// ops on flat vectors, no Tensor here

std::pair<std::vector<float>, std::vector<int>> _elementwise(
    const std::vector<float>& a_data, const std::vector<int>& a_shape,
    const std::vector<float>& b_data, const std::vector<int>& b_shape,
    std::function<float(float, float)> op);

template <typename Op>
std::vector<float> elemwise(const std::vector<float>& A, const std::vector<float>& B, int n, int ndim,
    const std::vector<int>& A_shape, const std::vector<int>& B_shape, const std::vector<int>& res_shape,
    const std::vector<int>& A_strides, const std::vector<int>& B_strides, Op op) {
        // UP-TO-DATE IMPLEMENTATION
        // requires res_shape, ndim and size of res_data

    std::vector<float> res_data(n);
    if (meta::same_shape(A_shape, B_shape)) {
        for (int i = 0; i < n; i++)
        res_data[i] = op(A[i], B[i]);
    } else if (A_shape.size() == 1 && A_shape[0] == 1) {
        for (int i = 0; i < n; i++)
        res_data[i] = op(A[0], B[i]);
    } else if (B_shape.size() == 1 && B_shape[0] == 1) {
        for (int i = 0; i < n; i++)
        res_data[i] = op(A[i], B[0]);
    } else {
        for (int i = 0; i < n; i++) {
            int a_idx = 0, b_idx = 0, remaining = i;
            for (int d = ndim - 1; d >= 0; d--) {
                int coord = remaining % res_shape[d];
                remaining /= res_shape[d];
                a_idx += coord * A_strides[d];
                b_idx += coord * B_strides[d];
            }
            res_data[i] = op(A[a_idx], B[b_idx]);
        }
    }
    return res_data;
}

std::vector<float> elemwise_mul(std::vector<float> a, std::vector<float> b);

std::vector<float> elemwise_mul_exact_shape(std::vector<float> a, std::vector<float> b);


std::pair<std::vector<float>, std::vector<int>> matmul(
    const std::vector<float>& a_data, const std::vector<int>& a_shape, const std::vector<int>& a_strides,
    const std::vector<float>& b_data, const std::vector<int>& b_shape, const std::vector<int>& b_strides);

std::vector<float> exp(std::vector<float> data);

std::vector<float> pow(std::vector<float> data, float p);

std::vector<float> log(std::vector<float> data);

std::vector<float> matmul_blas(
   const std::vector<float>& a_data, const std::vector<float>& b_data, int batch, int M, int N, int K);

std::pair<std::vector<float>, std::vector<float>> matmul_blas_backward(
    const std::vector<float>& a_data, const std::vector<int>& a_shape,
    const std::vector<float>& b_data, const std::vector<int>& b_shape,
    const std::vector<float>& grad,   const std::vector<int>& out_shape);

std::vector<float> sum(std::vector<float> values, std::vector<int> shape,
                       std::vector<int> strides, int dim);

}
