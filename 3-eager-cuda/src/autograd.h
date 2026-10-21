// include only once
#pragma once

// imports
#include <vector>
#include <memory>

// namespaces
#include "./tensor_data.h"

namespace autograd {

struct Node {
    std::vector<std::shared_ptr<Node>> next_functions;
    virtual std::vector<std::vector<float>> apply(std::vector<float> grad) = 0;
    virtual ~Node() = default;
};

// reduce data (out_shape) to tgt by summing broadcast dims
inline std::vector<float> sum_to(
    const std::vector<float>& data,
    const std::vector<int>& out_shape,
    const std::vector<int>& tgt);

struct MulBackward : Node {
    std::shared_ptr<TensorData> t1, t2;
    std::vector<int> out_shape;

    MulBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::vector<int> os);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct MatmulBackward : Node {
    std::shared_ptr<TensorData> t1, t2, res;

    MatmulBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::shared_ptr<TensorData> c);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct MatmulBlasBackward : Node {
    std::shared_ptr<TensorData> t1, t2, res;

    MatmulBlasBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::shared_ptr<TensorData> c);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct AddBackward : Node {
    std::shared_ptr<TensorData> t1, t2;
    std::vector<int> out_shape;

    AddBackward(std::shared_ptr<TensorData> a, std::shared_ptr<TensorData> b, std::vector<int> os);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct SumBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    SumBackward(std::shared_ptr<TensorData> td, int d, bool kd);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct MeanBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    MeanBackward(std::shared_ptr<TensorData> td, int d, bool kd);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct LogBackward : Node {
    std::shared_ptr<TensorData> t;

    LogBackward(std::shared_ptr<TensorData> td);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct ReshapeBackward : Node {
    // Used by reshape, squeeze, unsqueeze, flatten
    ReshapeBackward();
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct ExpBackward : Node {
    std::shared_ptr<TensorData> out;

    ExpBackward(std::shared_ptr<TensorData> td);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct PowBackward : Node {
    std::shared_ptr<TensorData> t;
    float p;

    PowBackward(std::shared_ptr<TensorData> td, float num);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct MaxBackward : Node {
    std::shared_ptr<TensorData> t;
    int dim;
    bool keepdims;

    MaxBackward(std::shared_ptr<TensorData> td, int d, bool kd);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

struct AccumulateGrad : Node {
    std::shared_ptr<TensorData> impl;

    AccumulateGrad(std::shared_ptr<TensorData> td);
    std::vector<std::vector<float>> apply(std::vector<float> grad) override;
};

std::vector<Node*> topo_sort(Node* n);
void accumulate_grads(const std::vector<Node*>& topo, Node* root, const std::vector<float>& root_grad);

}
