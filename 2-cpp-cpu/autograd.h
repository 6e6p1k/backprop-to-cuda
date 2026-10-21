// include only once
#pragma once

// imports
#include <vector>
#include <memory>
#include <map>
#include <limits>

// namespaces
#include "./meta.h"
#include "./ops.h"
#include "./tensor_data.h"

namespace autograd {

struct Node {
    std::vector<std::shared_ptr<Node>> next_functions;
    virtual std::vector<std::vector<float>> apply(std::vector<float> grad) = 0;
};

// reduce data (out_shape) to tgt by summing broadcast dims
inline std::vector<float> sum_to(
    const std::vector<float>& data,
    const std::vector<int>& out_shape,
    const std::vector<int>& tgt)
{
    int ondim = out_shape.size();
    std::vector<int> padded(ondim, 1);
    for (int i = 0; i < (int)tgt.size(); i++)
        padded[ondim - tgt.size() + i] = tgt[i];
    int res_size = 1;
    for (int s : tgt) res_size *= s;
    auto tgt_strides = meta::compute_strides(tgt);
    std::vector<float> result(res_size, 0.0f);
    for (int i = 0; i < (int)data.size(); i++) {
        auto g_idxs = meta::idx_1d_to_nd(i, out_shape);
        std::vector<int> t_idxs(tgt.size());
        for (int d = 0; d < (int)tgt.size(); d++) {
            int od = d + (ondim - (int)tgt.size());
            t_idxs[d] = (padded[od] == 1) ? 0 : g_idxs[od];
        }
        result[meta::idx_nd_to_1d(t_idxs, tgt_strides)] += data[i];
    }
    return result;
}

struct MulBackward : Node {
    std::shared_ptr<TensorData> t1, t2;
    std::vector<int> out_shape;

    MulBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::vector<int> os)
        : t1(a), t2(b), out_shape(os) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        // grad_t1 = (grad * t2) broadcast to out_shape, then reduce to t1->shape
        auto g_t1 = ops::elementwise(grad, out_shape, t2->values, t2->shape,
                                     [](float a, float b){ return a * b; }).first;
        // grad_t2 = (grad * t1) broadcast to out_shape, then reduce to t2->shape
        auto g_t2 = ops::elementwise(grad, out_shape, t1->values, t1->shape,
                                     [](float a, float b){ return a * b; }).first;
        return {sum_to(g_t1, out_shape, t1->shape),
                sum_to(g_t2, out_shape, t2->shape)};
    }
};

struct MatmulBackward : Node {
    std::shared_ptr<TensorData> t1, t2, res;

    MatmulBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::shared_ptr<TensorData> c)
        : t1(a), t2(b), res(c) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        // grad_A = grad @ B^T
        auto [t2_T_shape, t2_T_strides] = meta::transpose(t2->shape, t2->strides);
        auto [grad_a, _1] = ops::matmul(grad, res->shape, res->strides,
                                        t2->values, t2_T_shape, t2_T_strides);
        // grad_B = A^T @ grad
        auto [t1_T_shape, t1_T_strides] = meta::transpose(t1->shape, t1->strides);
        auto [grad_b, _2] = ops::matmul(t1->values, t1_T_shape, t1_T_strides,
                                        grad, res->shape, res->strides);
        return {grad_a, grad_b};
    }
};

struct MatmulBlasBackward : Node {
    std::shared_ptr<TensorData> t1, t2, res;

    MatmulBlasBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::shared_ptr<TensorData> c)
        : t1(a), t2(b), res(c) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        auto [grad_a, grad_b] = ops::matmul_blas_backward(
            t1->values, t1->shape,
            t2->values, t2->shape,
            grad,        res->shape);
        return {grad_a, grad_b};
    }
};

struct AddBackward : Node {
    std::shared_ptr<TensorData> t1, t2;
    std::vector<int> out_shape;

    AddBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::vector<int> os)
        : t1(a), t2(b), out_shape(os) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        return {sum_to(grad, out_shape, t1->shape),
                sum_to(grad, out_shape, t2->shape)};
    }
};

struct SumBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    SumBackward(std::shared_ptr<TensorData> td, int d, bool kd)
        : t(td), dim(d), keepdims(kd) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        std::vector<float> grad_t(t->values.size());

        std::vector<int> grad_shape = t->shape;
        if (keepdims) {
            grad_shape[dim] = 1;
        } else {
            grad_shape.erase(grad_shape.begin() + dim);
        }
        auto grad_strides = meta::compute_strides(grad_shape);

        for (int i = 0; i < (int)t->values.size(); i++) {
            auto grad_idxs = meta::idx_1d_to_nd(i, t->shape);
            if (keepdims) {
                grad_idxs[dim] = 0;
            } else {
                grad_idxs.erase(grad_idxs.begin() + dim);
            }
            grad_t[i] = grad[meta::idx_nd_to_1d(grad_idxs, grad_strides)];
        }
        return {grad_t};
    }
};

struct MeanBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    MeanBackward(std::shared_ptr<TensorData> td, int d, bool kd)
        : t(td), dim(d), keepdims(kd) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        std::vector<float> grad_t(t->values.size());
        float N = t->shape[dim];

        std::vector<int> grad_shape = t->shape;
        if (keepdims) {
            grad_shape[dim] = 1;
        } else {
            grad_shape.erase(grad_shape.begin() + dim);
        }
        auto grad_strides = meta::compute_strides(grad_shape);

        for (int i = 0; i < (int)t->values.size(); i++) {
            auto grad_idxs = meta::idx_1d_to_nd(i, t->shape);
            if (keepdims) {
                grad_idxs[dim] = 0;
            } else {
                grad_idxs.erase(grad_idxs.begin() + dim);
            }
            grad_t[i] = grad[meta::idx_nd_to_1d(grad_idxs, grad_strides)] / N;
        }
        return {grad_t};
    }
};

struct LogBackward : Node {
    std::shared_ptr<TensorData> t;

    LogBackward(std::shared_ptr<TensorData> td) : t(td) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        float eps = 1e-8f;
        std::vector<float> grad_t(t->values.size());
        for (int i = 0; i < (int)t->values.size(); i++)
            grad_t[i] = grad[i] / (t->values[i] + eps);
        return {grad_t};
    }
};

struct ReshapeBackward : Node {
    // Used by reshape, squeeze, unsqueeze, flatten
    ReshapeBackward() {}
    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        return {grad};
    }
};

struct ExpBackward : Node {
    std::shared_ptr<TensorData> out;

    ExpBackward(std::shared_ptr<TensorData> td) : out(td) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        std::vector<float> grad_t = ops::elemwise_mul_exact_shape(out->values, grad);
        return {grad_t};
    }
};

struct PowBackward : Node {
    std::shared_ptr<TensorData> t;
    float p;

    PowBackward(std::shared_ptr<TensorData> td, float num) : t(td), p(num) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        std::vector<float> grad_t(t->values.size());
        for (int i = 0; i < int(grad_t.size()); i++)
            grad_t[i] = p * std::pow(t->values[i], (p-1)) * grad[i];
        return {grad_t};
    }
};

struct MaxBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    MaxBackward(std::shared_ptr<TensorData> td, int d, bool kd)
        : t(td), dim(d), keepdims(kd) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        std::vector<float> grad_t(t->values.size(), 0.0f);

        std::vector<int> out_shape = t->shape;
        if (keepdims) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }

        for (int i = 0; i < (int)grad.size(); i++) {
            auto out_idxs = meta::idx_1d_to_nd(i, out_shape);
            // find argmax along dim
            int argmax = 0;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < t->shape[dim]; j++) {
                auto in_idxs = out_idxs;
                if (keepdims) in_idxs[dim] = j;
                else          in_idxs.insert(in_idxs.begin() + dim, j);
                float val = t->values[meta::idx_nd_to_1d(in_idxs, t->strides)];
                if (val > max_val) { max_val = val; argmax = j; }
            }
            // route grad only to the max position
            auto in_idxs = out_idxs;
            if (keepdims) in_idxs[dim] = argmax;
            else          in_idxs.insert(in_idxs.begin() + dim, argmax);
            grad_t[meta::idx_nd_to_1d(in_idxs, t->strides)] += grad[i];
        }
        return {grad_t};
    }
};

struct AccumulateGrad : Node {
    std::shared_ptr<TensorData> impl;

    AccumulateGrad(std::shared_ptr<TensorData> td) : impl(td) {}

    std::vector<std::vector<float>> apply(std::vector<float> grad) override {
        impl->grad = grad;
        return {};
    }
};

std::vector<Node*> topo_sort(Node* n) {
    // topological sort
    std::vector<Node*> topo;
    std::set<Node*> visited;
    std::function<void(Node*)> build_topo = [&](Node* node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);
        for (auto& next : node->next_functions)
            build_topo(next.get());
        topo.push_back(node);
    };
    build_topo(n);
    return topo;
}

void accumulate_grads(const std::vector<Node*>& topo, Node* root, const std::vector<float>& root_grad) {
    std::map<Node*, std::vector<float>> grad_map;
    grad_map[root] = root_grad;

    for (auto it = topo.rbegin(); it != topo.rend(); it++) {
        Node* node = *it;
        auto input_grads = node->apply(grad_map[node]);
        for (int i = 0; i < (int)node->next_functions.size(); i++) {
            if (node->next_functions[i]) {
                auto key = node->next_functions[i].get();
                auto it2 = grad_map.find(key);
                if (it2 == grad_map.end()) {
                    grad_map[key] = input_grads[i];
                } else {
                    for (int j = 0; j < (int)it2->second.size(); j++)
                        it2->second[j] += input_grads[i][j];
                }
            }
        }
    }
}

}

