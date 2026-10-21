// imports
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <functional>
#include <string>

// namespaces
#include "tensor.h"
#include "autograd.h"
#include "cuda_ops.cuh"



Tensor::Tensor(std::vector<float> values, std::vector<int> shape,
    std::string device, bool requires_grad
) {
    td = std::make_shared<TensorData>();
    td->device = deviceFromString(device);

    if (td->device == Device::CPU) {
        td->values = values;
        td->strides = meta::compute_strides(shape);
    } else if (td->device == Device::CUDA) {
        // calculate size of tensor (faster)
        int size = 1;
        for (int s : shape) size *= s;

        vec_from_cpu_to_cuda(&td->cuda_data, values, size);
        vec_from_cpu_to_cuda(&td->cuda_strides, meta::compute_strides(shape), shape.size());
    } else {
        throw std::logic_error("Unknown device.");
    }

    td->shape = shape;
    this->requires_grad = requires_grad;
    if (requires_grad) {
        grad_fn = std::make_shared<autograd::AccumulateGrad>(td);
    }
}

// initialise tensor filled with number
Tensor::Tensor(std::vector<int> shape, float fill_val, std::string device, bool requires_grad
) {
    td = std::make_shared<TensorData>();
    int size = meta::numel(shape);
    td->device = deviceFromString(device);

    if (td->device == Device::CPU) {
        std::vector<float> values(size, fill_val);
        td->values = values;
        td->strides = meta::compute_strides(shape);
    }
    else if (td->device == Device::CUDA) {
        td->cuda_data = get_filled_vec(fill_val, size);
        vec_from_cpu_to_cuda(&td->cuda_strides, meta::compute_strides(shape), shape.size());
    }
    else {
        throw std::logic_error("Unknown device.");
    }

    td->shape = shape;
    this->requires_grad = requires_grad;
    if (requires_grad) {
        grad_fn = std::make_shared<autograd::AccumulateGrad>(td);
    }
}

int Tensor::numel() const {
    return meta::numel(td->shape);
}

Tensor Tensor::to_cuda() const {
    tensor_to_cuda(td.get());
    return *this;
}

Tensor Tensor::to_cpu() const {
    tensor_to_cpu(td.get());
    return *this;
}

std::vector<float> Tensor::to_host_truncated(int max_elements) const {
    int n = std::min(td->numel(), max_elements);
    if (td->device == Device::CPU) {
        return std::vector<float>(td->values.begin(), td->values.begin() + n);
    } else if (td->device == Device::CUDA) {
        return retrieve_first_n(td->cuda_data, n);
    } else {
        throw std::logic_error("Unknown device.");
    }

}

std::vector<float> Tensor::to_host_full() const {
    if (td->device == Device::CPU) {
        return td->values;
    }
    return retrieve_first_n(td->cuda_data, td->numel());
}

float Tensor::at(std::vector<int> idxs) const {
    return td->values[meta::idx_nd_to_1d(idxs, td->strides)];
}

void Tensor::print_values() const {
    int n = td->numel();
    std::vector<float> values(n);
    if (td->device == Device::CPU) {
        values = td->values;
    }
    else if (td->device == Device::CUDA) {
        values = retrieve_first_n(td->cuda_data, n);
    }
    else {
        throw std::logic_error("Unknown device.");
    }
    std::cout << "Values: ";
    for (float x : values) std::cout << x << " ";
    std::cout << std::endl;
}

void Tensor::print_shape() const {
    std::cout << "Shape: ";
    for (int x : td->shape) std::cout << x << " ";
    std::cout << std::endl;
}

Tensor Tensor::add(const Tensor& other) const {
    Tensor result = dispatch_elementwise<operators::Add>(other);

    if (requires_grad || other.requires_grad) {
        auto node = std::make_shared<autograd::AddBackward>(td, other.td, result.td->shape);
        node->next_functions = {grad_fn, other.grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    Tensor result = dispatch_elementwise<operators::Mul>(other);

    if (requires_grad || other.requires_grad) {
        auto node = std::make_shared<autograd::MulBackward>(td, other.td, result.td->shape);
        node->next_functions = {grad_fn, other.grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::sum(int dim, bool keepdims) const {
    Tensor result = dispatch_sum(dim, keepdims);
    if (requires_grad) {
        auto node = std::make_shared<autograd::SumBackward>(td, dim, keepdims);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::contiguous() const {
    if (td->strides == meta::compute_strides(td->shape)) return *this;
    std::vector<float> new_data(td->values.size());
    for (int i = 0; i < (int)td->values.size(); i++)
        new_data[i] = at(meta::idx_1d_to_nd(i, td->shape));
    return Tensor(new_data, td->shape);
}

Tensor Tensor::transpose() const {
    auto [new_shape, new_strides] = meta::transpose(td->shape, td->strides);
    return Tensor(td->values, new_shape, new_strides);
}

Tensor Tensor::mean(int dim, bool keepdims) const {
    if (dim < 0) dim += (int)td->shape.size();
    std::vector<int> res_shape(keepdims ? td->shape.size() : td->shape.size() - 1);
    if (keepdims) {
        for (int i = 0; i < (int)td->shape.size(); i++)
            res_shape[i] = (i == dim) ? 1 : td->shape[i];
    } else {
        int c = 0;
        for (int i = 0; i < (int)td->shape.size(); i++)
            if (i != dim) res_shape[c++] = td->shape[i];
    }

    size_t res_size = td->values.size() / td->shape[dim];
    float N = td->shape[dim];
    std::vector<float> result_data(res_size);

    for (int i = 0; i < (int)res_size; i++) {
        auto idxs = meta::idx_1d_to_nd(i, res_shape);
        float s = 0;
        for (int j = 0; j < td->shape[dim]; j++) {
            auto input_idxs = idxs;
            if (keepdims) input_idxs[dim] = j;
            else          input_idxs.insert(input_idxs.begin() + dim, j);
            s += td->values[meta::idx_nd_to_1d(input_idxs, td->strides)];
        }
        result_data[i] = s / N;
    }

    Tensor result(result_data, res_shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::MeanBackward>(td, dim, keepdims);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::max(int dim, bool keepdims) const {
    if (dim < 0) dim += (int)td->shape.size();
    std::vector<int> res_shape(keepdims ? td->shape.size() : td->shape.size() - 1);
    if (keepdims) {
        for (int i = 0; i < (int)td->shape.size(); i++)
            res_shape[i] = (i == dim) ? 1 : td->shape[i];
    } else {
        int c = 0;
        for (int i = 0; i < (int)td->shape.size(); i++)
            if (i != dim) res_shape[c++] = td->shape[i];
    }

    size_t res_size = td->values.size() / td->shape[dim];
    std::vector<float> result_data(res_size);

    for (int i = 0; i < (int)res_size; i++) {
        auto idxs = meta::idx_1d_to_nd(i, res_shape);
        float m = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < td->shape[dim]; j++) {
            auto input_idxs = idxs;
            if (keepdims) input_idxs[dim] = j;
            else          input_idxs.insert(input_idxs.begin() + dim, j);
            m = std::max(m, td->values[meta::idx_nd_to_1d(input_idxs, td->strides)]);
        }
        result_data[i] = m;
    }

    Tensor result(result_data, res_shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::MaxBackward>(td, dim, keepdims);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::argmax(int dim) const {
    if (dim < 0) dim += (int)td->shape.size();

    std::vector<int> res_shape;
    res_shape.reserve(td->shape.size() - 1);
    for (int i = 0; i < (int)td->shape.size(); i++)
        if (i != dim) res_shape.push_back(td->shape[i]);

    size_t res_size = 1;
    for (int s : res_shape) res_size *= s;

    std::vector<float> result_data(res_size);
    std::vector<int> input_idxs(td->shape.size());

    for (size_t i = 0; i < res_size; i++) {
        auto res_idxs = meta::idx_1d_to_nd((int)i, res_shape);

        int ri = 0;
        for (int d = 0; d < (int)td->shape.size(); d++)
            input_idxs[d] = (d == dim) ? 0 : res_idxs[ri++];

        int best = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < td->shape[dim]; j++) {
            input_idxs[dim] = j;
            float val = td->values[meta::idx_nd_to_1d(input_idxs, td->strides)];
            if (val > best_val) { best_val = val; best = j; }
        }

        result_data[i] = (float)best;
    }

    return Tensor(result_data, res_shape);
}

Tensor Tensor::eq(const Tensor& other) const {
    // rn only for equal size
    if (td->values.size() != other.td->values.size()) throw std::logic_error("Tensors should have equal sizes.");
    std::vector<float> result_data(td->values.size());
    for (int i = 0; i < (int)td->values.size(); i++)
    if (td->values[i] == other.td->values[i]) result_data[i] = (float)1;
    else result_data[i] = (float)0;
    return Tensor(result_data, td->shape);
}

Tensor Tensor::log() const {
    std::vector<float> result_data = ops::log(td->values);
    Tensor result(result_data, td->shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::LogBackward>(td);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    int new_size = 1;
    for (int s : new_shape) new_size *= s;
    if (new_size != (int)td->values.size()) throw std::logic_error("Shapes don't match.");
    Tensor result(td->values, new_shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::ReshapeBackward>();
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += (int)td->shape.size();
    Tensor result(td->values, meta::squeeze_shape(td->shape, dim));
    if (requires_grad) {
        auto node = std::make_shared<autograd::ReshapeBackward>();
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += (int)td->shape.size() + 1;  // +1: unsqueeze can insert after last dim
    Tensor result(td->values, meta::unsqueeze_shape(td->shape, dim));
    if (requires_grad) {
        auto node = std::make_shared<autograd::ReshapeBackward>();
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    Tensor result(td->values, meta::flatten(td->shape, start_dim, end_dim));
    if (requires_grad) {
        auto node = std::make_shared<autograd::ReshapeBackward>();
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    meta::check_matmul_shapes(td->shape, other.td->shape);
    auto [result_data, res_shape] = ops::matmul(td->values, td->shape, td->strides,
                                                    other.td->values, other.td->shape, other.td->strides);
    Tensor result(result_data, res_shape);
    if (requires_grad || other.requires_grad) {
        auto node = std::make_shared<autograd::MatmulBackward>(td, other.td, result.td);
        node->next_functions = {grad_fn, other.grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::matmul_fast(const Tensor& other) const {
    if (td->device != other.td->device)
        throw std::logic_error("Tensors are on different devices.");
    meta::check_matmul_shapes(td->shape, other.td->shape);

    Tensor result = dispatch_matmul(other);

    if (requires_grad || other.requires_grad) {
        auto node = std::make_shared<autograd::MatmulBlasBackward>(td, other.td, result.td);
        node->next_functions = {grad_fn, other.grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::matmul_blas(const Tensor& other) const {
    meta::check_matmul_shapes(td->shape, other.td->shape);
    auto dims = meta::get_matmul_dims(td->shape, other.td->shape);
    auto res_data = ops::matmul_blas(td->values, other.td->values,
        dims.batch, dims.M, dims.N, dims.K);
    return Tensor(res_data, dims.res_shape);
}

Tensor Tensor::matmul_cuda(const Tensor& other) const {
    meta::check_matmul_shapes(td->shape, other.td->shape);
    auto dims = meta::get_matmul_dims(td->shape, other.td->shape);

    std::shared_ptr<TensorData> a_tmp, b_tmp;
    const float* a_gpu;
    const float* b_gpu;

    if (td->device == Device::CUDA) {
        a_gpu = td->cuda_data;
    } else {
        a_tmp = std::make_shared<TensorData>();
        vec_from_cpu_to_cuda(&a_tmp->cuda_data, td->values, meta::numel(td->shape));
        a_gpu = a_tmp->cuda_data;
    }

    if (other.td->device == Device::CUDA) {
        b_gpu = other.td->cuda_data;
    } else {
        b_tmp = std::make_shared<TensorData>();
        vec_from_cpu_to_cuda(&b_tmp->cuda_data, other.td->values, meta::numel(other.td->shape));
        b_gpu = b_tmp->cuda_data;
    }

    float* ptr = cuda_ops::matmul_cuda_launch(a_gpu, b_gpu, dims.batch, dims.M, dims.N, dims.K);
    return Tensor(ptr, dims.res_shape);
}

Tensor Tensor::exp() const {
    std::vector<float> result_data = ops::exp(td->values);
    Tensor result(result_data, td->shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::ExpBackward>(result.td);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

Tensor Tensor::pow(const float num) const {
    std::vector<float> result_data = ops::pow(td->values, num);
    Tensor result(result_data, td->shape);
    if (requires_grad) {
        auto node = std::make_shared<autograd::PowBackward>(td, num);
        node->next_functions = {grad_fn};
        result.grad_fn = node;
        result.requires_grad = true;
    }
    return result;
}

void Tensor::reset_grad() {
    td->grad.clear();
}

void Tensor::sgd_update(float lr) {
    if (td->values.size() != td->grad.size()) throw std::logic_error("Sizes of grad and tensor are different!");
    for (int i = 0; i < (int)td->values.size(); i++)
        td->values[i] -= lr * td->grad[i];
}

void Tensor::backward() {
    // init gradient of output to 1
    td->grad = std::vector<float>(td->values.size(), 1.0f);

    // topological sort
    std::vector<autograd::Node*> topo = autograd::topo_sort(grad_fn.get());

    // reverse traversal, accumulate grads
    autograd::accumulate_grads(topo, grad_fn.get(), td->grad);
}


Tensor::Tensor(std::vector<float> values, std::vector<int> shape, std::vector<int> strides) {
    td = std::make_shared<TensorData>();
    td->values = values;
    td->shape = shape;
    td->strides = strides;
}

Tensor::Tensor(float* cuda_ptr, std::vector<int> shape) {
    td = std::make_shared<TensorData>();
    td->cuda_data = cuda_ptr;
    td->device = Device::CUDA;
    td->shape = shape;
    vec_from_cpu_to_cuda(&td->cuda_strides, meta::compute_strides(shape), shape.size());
}

Tensor::Tensor(float* cuda_ptr, std::vector<int> shape, int* strides) {
    td = std::make_shared<TensorData>();
    td->cuda_data = cuda_ptr;
    td->device = Device::CUDA;
    td->shape = shape;
    td->cuda_strides = strides;
}

Tensor Tensor::dispatch_matmul(const Tensor& other) const {
    auto dims = meta::get_matmul_dims(td->shape, other.td->shape);
    if (td->device == Device::CPU) {
        auto res_data = ops::matmul_blas(td->values, other.td->values,
            dims.batch, dims.M, dims.N, dims.K);
        return Tensor(res_data, dims.res_shape);
    } else if (td->device == Device::CUDA) {
        float* ptr = cuda_ops::matmul_cuda_launch(td->cuda_data, other.td->cuda_data,
            dims.batch, dims.M, dims.N, dims.K);
        return Tensor(ptr, dims.res_shape);
    }
    throw std::logic_error("Unknown device.");
}

Tensor Tensor::dispatch_sum(int dim, bool keepdims) const {
    if (dim < 0) dim += (int)td->shape.size();
    std::vector<int> res_shape(keepdims ? td->shape.size() : td->shape.size() - 1);
    if (keepdims) {
        for (int i = 0; i < (int)td->shape.size(); i++)
            res_shape[i] = (i == dim) ? 1 : td->shape[i];
    } else {
        int c = 0;
        for (int i = 0; i < (int)td->shape.size(); i++)
            if (i != dim) res_shape[c++] = td->shape[i];
    }

    if (td->device == Device::CPU) {
        auto result_data = ops::sum(td->values, td->shape, td->strides, dim);
        return Tensor(result_data, res_shape);
    } else if (td->device == Device::CUDA) {
        float* ptr = cuda_ops::sum_launch(td->cuda_data, td->shape, td->cuda_strides, dim);
        return Tensor(ptr, res_shape);
    } else {
        throw std::logic_error("Unknown device.");
    }
}





