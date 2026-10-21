#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cutensor.h>
#include <numeric>
#include <cstdio>

#include "cuda_ops.cuh"
#include "tensor_data.h"
#include "meta.h"


// TensordData destrcutor
TensorData::~TensorData() {
    if (cuda_data != nullptr) {
        cudaFree(cuda_data);
        cudaFree(cuda_strides);
        cuda_data = nullptr;
        cuda_strides = nullptr;
    }
}

std::vector<float> retrieve_first_n(const float* cuda_data, int n) {
    std::vector<float> result(n);
    cudaMemcpy(result.data(), cuda_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

template <typename T>
void vec_from_cpu_to_cuda(T** cuda_data, const std::vector<T>& values, int n) {
    cudaMalloc(cuda_data, n * sizeof(T));
    cudaMemcpy(*cuda_data, values.data(), n * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void vec_from_cuda_to_cpu(T* values_ptr, T* cuda_values, int n) {
    cudaMemcpy(values_ptr, cuda_values, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(cuda_values);
}

void tensor_to_cuda(TensorData* td) {
    if (td->device == Device::CUDA) return;
    int n = td->numel();
    vec_from_cpu_to_cuda(&td->cuda_data, td->values, n);
    vec_from_cpu_to_cuda(&td->cuda_strides, td->strides, td->strides.size());
    td->device = Device::CUDA;
}

void tensor_to_cpu(TensorData* td) {
    if (td->device == Device::CPU) return;
    int n = td->numel();
    td->values.resize(n);
    td->strides.resize(td->shape.size());
    vec_from_cuda_to_cpu(td->values.data(), td->cuda_data, n);
    vec_from_cuda_to_cpu(td->strides.data(), td->cuda_strides, td->shape.size());
    td->cuda_data = nullptr;
    td->cuda_strides = nullptr;
    td->device = Device::CPU;
}

namespace cuda_ops {


float* matmul_cuda_launch(
    const float* A, const float* B, int batch, int M, int N, int K)
{
    // created once, reused for every call
    static cublasHandle_t handle = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

        // allocate 32MB workspace
        void* workspace;
        cudaMalloc(&workspace, 32 * 1024 * 1024);
        cublasSetWorkspace(h, workspace, 32 * 1024 * 1024);

        return h;
    }();

    float* res_ptr;
    cudaMalloc(&res_ptr, batch * M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N, K * N,
        A, CUDA_R_32F, K, M * K,
        &beta,
        res_ptr, CUDA_R_32F, N, M * N,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    return res_ptr;
}

// </MATMUL>





// <ELEMWISE>

#define MAX_DIMS 8

struct BroadcastInfo {
    int out_shape[MAX_DIMS];
    int a_strides[MAX_DIMS];   // 0 for broadcast dims
    int b_strides[MAX_DIMS];   // 0 for broadcast dims
    int ndim;
};

template <typename Op>
__global__ void broadcast_elemwise_kernel(
    const float* A, const float* B, float* C,
    BroadcastInfo info, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int remaining = idx;
    int a_idx = 0, b_idx = 0;
    for (int d = info.ndim - 1; d >= 0; d--) {
        int coord = remaining % info.out_shape[d];
        remaining /= info.out_shape[d];
        a_idx += coord * info.a_strides[d];
        b_idx += coord * info.b_strides[d];
    }
    C[idx] = Op{}(A[a_idx], B[b_idx]);
}

template <typename Op>
static float* elemwise_broadcast(
    const float* A, const float* B,
    const std::vector<int>& A_shape,
    const std::vector<int>& B_shape,
    const std::vector<int>& out_shape)
{
    int out_ndim = out_shape.size();
    int out_numel = 1;
    for (int s : out_shape) out_numel *= s;

    BroadcastInfo info;
    info.ndim = out_ndim;

    auto compute_broadcast_strides = [&](const std::vector<int>& shape, int* strides) {
        int off = out_ndim - (int)shape.size();
        for (int i = 0; i < off; i++) strides[i] = 0;
        int s = 1;
        for (int i = (int)shape.size() - 1; i >= 0; i--) {
            strides[off + i] = (shape[i] == 1) ? 0 : s;
            s *= shape[i];
        }
    };

    for (int i = 0; i < out_ndim; i++) info.out_shape[i] = out_shape[i];
    compute_broadcast_strides(A_shape, info.a_strides);
    compute_broadcast_strides(B_shape, info.b_strides);

    float* res_ptr;
    cudaMalloc(&res_ptr, out_numel * sizeof(float));

    int threads = 256;
    int blocks = (out_numel + threads - 1) / threads;
    broadcast_elemwise_kernel<Op><<<blocks, threads>>>(A, B, res_ptr, info, out_numel);
    return res_ptr;
}

template <typename Op>
float* elemwise_launch(const float* A, const float* B, int n, int ndim,
    const std::vector<int>& A_shape, const std::vector<int>& B_shape,
    const std::vector<int>& out_shape, int* A_strides, int* B_strides)
{
    return elemwise_broadcast<Op>(A, B, A_shape, B_shape, out_shape);
}
// </ELEMWISE>



// <REDUCE>
float* cuda_reduce_wrapper(const float* A, const std::vector<int>& A_shape, int* A_strides,
    int dim, cutensorOperator_t Op)
{
    static cutensorHandle_t handle = [](){
        cutensorHandle_t h;
        cutensorCreate(&h);
        return h;
    }();

    int ndim = A_shape.size();

    // strides are on gpu, copy to cpu for the descriptor
    std::vector<int> strides_cpu(ndim);
    cudaMemcpy(strides_cpu.data(), A_strides, ndim * sizeof(int), cudaMemcpyDeviceToHost);

    // modes: 0..ndim-1 for input, same but with dim dropped for output
    std::vector<int32_t> modesA(ndim);
    std::iota(modesA.begin(), modesA.end(), 0);
    std::vector<int32_t> modesC;
    for (int i = 0; i < ndim; i++)
        if (i != dim) modesC.push_back(i);

    // input extents and strides
    std::vector<int64_t> extentA(A_shape.begin(), A_shape.end());
    std::vector<int64_t> stridesA(strides_cpu.begin(), strides_cpu.end());

    // output extents and contiguous strides (dim axis dropped)
    int out_ndim = ndim - 1;
    std::vector<int64_t> extentC(out_ndim), stridesC(out_ndim);
    int out_numel = 1;
    for (int i = out_ndim - 1; i >= 0; i--) {
        extentC[i] = A_shape[modesC[i]];
        stridesC[i] = out_numel;
        out_numel  *= A_shape[modesC[i]];
    }

    cutensorTensorDescriptor_t descA, descC;
    cutensorCreateTensorDescriptor(handle, &descA, ndim,     extentA.data(), stridesA.data(), CUTENSOR_R_32F, 128);
    cutensorCreateTensorDescriptor(handle, &descC, out_ndim, extentC.data(), stridesC.data(), CUTENSOR_R_32F, 128);

    float* res_ptr;
    cudaMalloc(&res_ptr, out_numel * sizeof(float));

    cutensorOperationDescriptor_t opDesc;
    cutensorCreateReduction(handle, &opDesc,
        descA, modesA.data(), CUTENSOR_OP_IDENTITY,
        descC, modesC.data(), CUTENSOR_OP_IDENTITY,
        descC, modesC.data(),
        Op, CUTENSOR_COMPUTE_DESC_32F);

    cutensorPlanPreference_t planPref;
    cutensorCreatePlanPreference(handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);

    uint64_t workspaceSize;
    cutensorEstimateWorkspaceSize(handle, opDesc, planPref, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSize);

    static void* workspace = nullptr;
    static uint64_t workspaceAllocated = 0;
    if (workspaceSize > workspaceAllocated) {
        if (workspace) cudaFree(workspace);
        cudaMalloc(&workspace, workspaceSize);
        workspaceAllocated = workspaceSize;
    }

    cutensorPlan_t plan;
    cutensorCreatePlan(handle, &plan, opDesc, planPref, workspaceSize);

    float alpha = 1.0f, beta = 0.0f;
    cutensorReduce(handle, plan, &alpha, A, &beta, res_ptr, res_ptr, workspace, workspaceSize, 0);

    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(planPref);
    cutensorDestroyOperationDescriptor(opDesc);
    cutensorDestroyTensorDescriptor(descA);
    cutensorDestroyTensorDescriptor(descC);
    return res_ptr;
}
// </REDUCE>

// <SUM>
float* sum_launch(const float* A, const std::vector<int>& A_shape, int* A_strides, int dim) {
    return cuda_reduce_wrapper(A, A_shape, A_strides, dim, CUTENSOR_OP_ADD);
}
// </SUM>

// <MAX>
float* max_launch(const float* A, const std::vector<int>& A_shape, int* A_strides, int dim) {
    return cuda_reduce_wrapper(A, A_shape, A_strides, dim, CUTENSOR_OP_MAX);
}
// </MAX>

// <MIN>
float* min_launch(const float* A, const std::vector<int>& A_shape, int* A_strides, int dim) {
    return cuda_reduce_wrapper(A, A_shape, A_strides, dim, CUTENSOR_OP_MIN);
}
// </MIN>


}

// <FILL_KERNEL>
__global__ void fill_kernel(float* data, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}
// </FILL_KERNEL>

float* get_filled_vec(float fill_val, int size) {
    float* cuda_data;
    cudaMalloc(&cuda_data, size * sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    fill_kernel<<<blocks, threads>>>(cuda_data, fill_val, size);
    cudaDeviceSynchronize();
    return cuda_data;
}


template float* cuda_ops::elemwise_launch<operators::Add>(const float*, const float*, int, int, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, int*, int*);
template float* cuda_ops::elemwise_launch<operators::Mul>(const float*, const float*, int, int, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, int*, int*);
template float* cuda_ops::elemwise_launch<operators::Div>(const float*, const float*, int, int, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, int*, int*);
template float* cuda_ops::elemwise_launch<operators::Sub>(const float*, const float*, int, int, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, int*, int*);

template void vec_from_cpu_to_cuda<float>(float**, const std::vector<float>&, int);
template void vec_from_cpu_to_cuda<int>(int**, const std::vector<int>&, int);

template void vec_from_cuda_to_cpu<float>(float*, float*, int);
template void vec_from_cuda_to_cpu<int>(int*, int*, int);
