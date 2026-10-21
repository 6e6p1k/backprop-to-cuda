#pragma once

#include <vector>
#include <cutensor.h>

#include "tensor_data.h"


void tensor_to_cuda(TensorData* td);
void tensor_to_cpu(TensorData* td);

template <typename T>
void vec_from_cpu_to_cuda(T** cuda_data, const std::vector<T>& values, int n);
template <typename T>
void vec_from_cuda_to_cpu(T* values_ptr, T* cuda_values, int n);

std::vector<float> retrieve_first_n(const float* cuda_data, int n);

namespace cuda_ops {
float* matmul_cuda_launch(const float* A, const float* B, int batch, int M, int N, int K);

template <typename Op>
float* elemwise_launch(const float* A, const float* B, int n, int ndim,
    const std::vector<int>& A_shape, const std::vector<int>& B_shape,
    const std::vector<int>& out_shape, int* A_strides, int* B_strides);

float* sum_launch(const float* A, const std::vector<int>& A_shape, int* A_strides, int dim);

}

float* get_filled_vec(float fill_val, int size);

