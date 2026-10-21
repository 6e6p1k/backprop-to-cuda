// include only once
#pragma once

#include <vector>
#include <stdexcept>
#include <functional>
#include "./meta.h"
#include "cblas.h"

namespace ops {
// ops on flat vectors, no Tensor here

std::pair<std::vector<float>, std::vector<int>> elementwise(
    const std::vector<float>& a_data, const std::vector<int>& a_shape,
    const std::vector<float>& b_data, const std::vector<int>& b_shape,
    std::function<float(float, float)> op)
{
    int ndim = std::max(a_shape.size(), b_shape.size());
    std::vector<int> bs1(ndim, 1), bs2(ndim, 1), res_shape(ndim);

    for (int i = 0; i < (int)a_shape.size(); i++) bs1[ndim - a_shape.size() + i] = a_shape[i];
    for (int i = 0; i < (int)b_shape.size(); i++) bs2[ndim - b_shape.size() + i] = b_shape[i];

    for (int i = 0; i < ndim; i++) {
        if (bs1[i] != bs2[i] && bs1[i] != 1 && bs2[i] != 1)
            throw std::logic_error("Shape mismatch, not broadcastable.");
        res_shape[i] = std::max(bs1[i], bs2[i]);
    }

    auto st1 = meta::compute_strides(bs1);
    auto st2 = meta::compute_strides(bs2);

    size_t res_size = 1;
    for (int s : res_shape) res_size *= s;
    std::vector<float> result_data(res_size);

    for (int i = 0; i < (int)res_size; i++) {
        auto idxs = meta::idx_1d_to_nd(i, res_shape);
        auto idx1 = idxs, idx2 = idxs;
        for (int d = 0; d < ndim; d++) {
            if (bs1[d] == 1) idx1[d] = 0;
            if (bs2[d] == 1) idx2[d] = 0;
        }
        result_data[i] = op(
            a_data[meta::idx_nd_to_1d(idx1, st1)],
            b_data[meta::idx_nd_to_1d(idx2, st2)]
        );
    }

    return {result_data, res_shape};
}

std::vector<float> elemwise_mul(std::vector<float> a, std::vector<float> b) {
    if (a.size() == b.size()) {
        std::vector<float> result(a.size());
        for (int i = 0; i < (int)a.size(); i++)
            result[i] = a[i] * b[i];
        return result;
    }
    if (a.size() == 1) {
        std::vector<float> result(b.size());
        for (int i = 0; i < (int)b.size(); i++)
            result[i] = a[0] * b[i];
        return result;
    }
    if (b.size() == 1) {
        std::vector<float> result(a.size());
        for (int i = 0; i < (int)a.size(); i++)
            result[i] = a[i] * b[0];
        return result;
    }
    throw std::logic_error("shape mismatch");
}

std::vector<float> elemwise_mul_exact_shape(std::vector<float> a, std::vector<float> b) {
    if (a.size() != b.size()) throw std::logic_error("shape mismatch");
    std::vector<float> result(a.size());
    for (int i = 0; i < (int)a.size(); i++)
        result[i] = a[i] * b[i];
    return result;
}

std::pair<std::vector<float>, std::vector<int>> matmul(
    const std::vector<float>& a_data, const std::vector<int>& a_shape, const std::vector<int>& a_strides,
    const std::vector<float>& b_data, const std::vector<int>& b_shape, const std::vector<int>& b_strides)
{
    int m = a_shape[a_shape.size() - 2];
    int k = a_shape[a_shape.size() - 1];
    int n = b_shape[b_shape.size() - 1];
    int ndim = a_shape.size();

    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) batch *= a_shape[i];

    std::vector<float> result_data(batch * m * n, 0.0f);

    for (int b = 0; b < batch; b++) {
        int a_batch_offset = 0, b_batch_offset = 0, tmp = b;
        for (int d = ndim - 3; d >= 0; d--) {
            a_batch_offset += (tmp % a_shape[d]) * a_strides[d];
            b_batch_offset += (tmp % b_shape[d]) * b_strides[d];
            tmp /= a_shape[d];
        }
        int r_offset = b * m * n;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int p = 0; p < k; p++) {
                    result_data[r_offset + i*n + j] +=
                        a_data[a_batch_offset + i*a_strides[ndim-2] + p*a_strides[ndim-1]] *
                        b_data[b_batch_offset + p*b_strides[ndim-2] + j*b_strides[ndim-1]];
                }
            }
        }
    }

    std::vector<int> res_shape(a_shape.begin(), a_shape.end() - 2);
    res_shape.push_back(m);
    res_shape.push_back(n);

    return {result_data, res_shape};
}

std::vector<float> exp(std::vector<float> data) {
    std::vector<float> out(data.size());
    for (int i = 0; i < (int)data.size(); i++)
    out[i] = std::exp(data[i]);
    return out;
}

std::vector<float> pow(std::vector<float> data, float p) {
    std::vector<float> out(data.size());
    for (int i = 0; i < (int)data.size(); i++) {
        out[i] = std::pow(data[i], p);
    }
    return out;
}

std::vector<float> log(std::vector<float> data) {
    float eps = 1e-8f;
    std::vector<float> out(data.size());
    for (int i = 0; i < (int)data.size(); i++)
        out[i] = std::log(data[i] + eps);
    return out;
}

// blas matmul, inputs must be contiguous row-major, no transposes
std::pair<std::vector<float>, std::vector<int>> matmul_blas(
    const std::vector<float>& a_data, const std::vector<int>& a_shape,
    const std::vector<float>& b_data, const std::vector<int>& b_shape)
{
    int ndim = a_shape.size();
    int M = a_shape[ndim - 2];
    int K = a_shape[ndim - 1];
    int N = b_shape[ndim - 1];
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) batch *= a_shape[i];

    std::vector<float> result(batch * M * N, 0.0f);

    for (int b = 0; b < batch; b++) {
        scipy_cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0f,
            a_data.data() + b * M * K, K,
            b_data.data() + b * K * N, N,
            0.0f,
            result.data() + b * M * N, N
        );
    }

    std::vector<int> res_shape(a_shape.begin(), a_shape.end() - 2);
    res_shape.push_back(M);
    res_shape.push_back(N);
    return {result, res_shape};
}

// grad_A = grad @ B^T,  grad_B = A^T @ grad
std::pair<std::vector<float>, std::vector<float>> matmul_blas_backward(
    const std::vector<float>& a_data, const std::vector<int>& a_shape,
    const std::vector<float>& b_data, const std::vector<int>& b_shape,
    const std::vector<float>& grad,   const std::vector<int>& out_shape)
{
    int ndim  = out_shape.size();
    int M     = out_shape[ndim - 2];
    int N     = out_shape[ndim - 1];
    int K     = a_shape[a_shape.size() - 1];
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) batch *= out_shape[i];

    std::vector<float> grad_a(batch * M * K, 0.0f);
    std::vector<float> grad_b(batch * K * N, 0.0f);

    for (int b = 0; b < batch; b++) {
        // grad_A = grad @ B^T, (M,N) x (K,N)^T = (M,K)
        scipy_cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            M, K, N, 1.0f,
            grad.data()   + b * M * N, N,
            b_data.data() + b * K * N, N,
            0.0f,
            grad_a.data() + b * M * K, K
        );
        // grad_B = A^T @ grad, (M,K)^T x (M,N) = (K,N)
        scipy_cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            K, N, M, 1.0f,
            a_data.data() + b * M * K, K,
            grad.data()   + b * M * N, N,
            0.0f,
            grad_b.data() + b * K * N, N
        );
    }

    return {grad_a, grad_b};
}

}