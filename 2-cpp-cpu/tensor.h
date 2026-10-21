// include only once
#pragma once

// imports
#include <iostream>
#include <vector>
#include <memory>
#include <functional>

// namespaces
#include "./autograd.h"
#include "./ops.h"
#include "./meta.h"
#include "./tensor_data.h"



// tensor
class Tensor {
public:
    std::shared_ptr<TensorData> td;
    std::shared_ptr<autograd::Node> grad_fn;
    bool requires_grad = false;

    Tensor(std::vector<float> values, std::vector<int> shape, bool requires_grad = false) {
        td = std::make_shared<TensorData>();
        td->values = values;
        td->shape = shape;
        td->strides = meta::compute_strides(shape);
        this->requires_grad = requires_grad;
        if (requires_grad) {
            grad_fn = std::make_shared<autograd::AccumulateGrad>(td);
        }
    }

    int numel() const {
        int n = 1;
        for (int s : td->shape) n *= s;
        return n;
    }

    float at(std::vector<int> idxs) const {
        return td->values[meta::idx_nd_to_1d(idxs, td->strides)];
    }

    void print() const {
        for (float x : td->values) std::cout << x << " ";
        std::cout << std::endl;
    }

    void print_shape() const {
        std::cout << "Shape: ";
        for (int x : td->shape) std::cout << x << " ";
        std::cout << std::endl;
    }

    Tensor elementwise(const Tensor& other, std::function<float(float, float)> op) const {
        auto [result_data, res_shape] = ops::elementwise(
            td->values, td->shape, other.td->values, other.td->shape, op);
        return Tensor(result_data, res_shape);
    }

    Tensor add(const Tensor& other) const {
        Tensor result = elementwise(other, [](float a, float b) { return a + b; });
        if (requires_grad || other.requires_grad) {
            auto node = std::make_shared<autograd::AddBackward>(td, other.td, result.td->shape);
            node->next_functions = {grad_fn, other.grad_fn};
            result.grad_fn = node;
            result.requires_grad = true;
        }
        return result;
    }

    Tensor mul(const Tensor& other) const {
        Tensor result = elementwise(other, [](float a, float b) { return a * b; });
        if (requires_grad || other.requires_grad) {
            auto node = std::make_shared<autograd::MulBackward>(td, other.td, result.td->shape);
            node->next_functions = {grad_fn, other.grad_fn};
            result.grad_fn = node;
            result.requires_grad = true;
        }
        return result;
    }

    Tensor sum(int dim, bool keepdims) const {
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
            float s = 0;
            for (int j = 0; j < td->shape[dim]; j++) {
                auto input_idxs = idxs;
                if (keepdims) input_idxs[dim] = j;
                else          input_idxs.insert(input_idxs.begin() + dim, j);
                s += td->values[meta::idx_nd_to_1d(input_idxs, td->strides)];
            }
            result_data[i] = s;
        }

        Tensor result(result_data, res_shape);
        if (requires_grad) {
            auto node = std::make_shared<autograd::SumBackward>(td, dim, keepdims);
            node->next_functions = {grad_fn};
            result.grad_fn = node;
            result.requires_grad = true;
        }
        return result;
    }

    Tensor contiguous() const {
        if (td->strides == meta::compute_strides(td->shape)) return *this;
        std::vector<float> new_data(td->values.size());
        for (int i = 0; i < (int)td->values.size(); i++)
            new_data[i] = at(meta::idx_1d_to_nd(i, td->shape));
        return Tensor(new_data, td->shape);
    }

    Tensor transpose() const {
        auto [new_shape, new_strides] = meta::transpose(td->shape, td->strides);
        return Tensor(td->values, new_shape, new_strides);
    }

    Tensor mean(int dim, bool keepdims) const {
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

    Tensor max(int dim, bool keepdims) const {
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

    Tensor argmax(int dim) const {
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

    Tensor eq(const Tensor& other) const {
        // rn only for equal size
        if (td->values.size() != other.td->values.size()) throw std::logic_error("Tensors should have equal sizes.");
        std::vector<float> result_data(td->values.size());
        for (int i = 0; i < (int)td->values.size(); i++) 
        if (td->values[i] == other.td->values[i]) result_data[i] = (float)1;
        else result_data[i] = (float)0;
        return Tensor(result_data, td->shape);
    }

    Tensor log() const {
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

    Tensor reshape(std::vector<int> new_shape) const {
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

    Tensor squeeze(int dim) const {
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

    Tensor unsqueeze(int dim) const {
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

    Tensor flatten(int start_dim, int end_dim) const {
        Tensor result(td->values, meta::flatten(td->shape, start_dim, end_dim));
        if (requires_grad) {
            auto node = std::make_shared<autograd::ReshapeBackward>();
            node->next_functions = {grad_fn};
            result.grad_fn = node;
            result.requires_grad = true;
        }
        return result;
    }

    Tensor matmul(const Tensor& other) const {
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

    Tensor matmul_blas(const Tensor& other) const {
        meta::check_matmul_shapes(td->shape, other.td->shape);
        auto [result_data, res_shape] = ops::matmul_blas(td->values, td->shape,
                                                          other.td->values, other.td->shape);
        Tensor result(result_data, res_shape);
        if (requires_grad || other.requires_grad) {
            auto node = std::make_shared<autograd::MatmulBlasBackward>(td, other.td, result.td);
            node->next_functions = {grad_fn, other.grad_fn};
            result.grad_fn = node;
            result.requires_grad = true;
        }
        return result;
    }

    Tensor exp() const {
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

    Tensor pow(const float num) const {
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

    void reset_grad() {
        td->grad.clear();
    }

    void update(float lr) {
        if (td->values.size() != td->grad.size()) throw std::logic_error("Sizes of grad and tensor are different!");
        for (int i = 0; i < (int)td->values.size(); i++)
            td->values[i] -= lr * td->grad[i];
    }

    void backward() {
        // init gradient of output to 1
        td->grad = std::vector<float>(td->values.size(), 1.0f);

        // topological sort
        std::vector<autograd::Node*> topo = autograd::topo_sort(grad_fn.get());

        // reverse traversal, accumulate grads
        autograd::accumulate_grads(topo, grad_fn.get(), td->grad);
    }

private:
    Tensor(std::vector<float> values, std::vector<int> shape, std::vector<int> strides) {
        td = std::make_shared<TensorData>();
        td->values = values;
        td->shape = shape;
        td->strides = strides;
    }
};

