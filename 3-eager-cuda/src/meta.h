#pragma once

#include <vector>
#include <stdexcept>


namespace meta {
// shape/stride/index stuff only, no data

inline int numel(const std::vector<int> shape) {
    int n = 1;
    for (int s : shape) n *= s;
    return n;
}

inline std::vector<int> compute_strides(std::vector<int> shape) {
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

inline std::pair<std::vector<int>, std::vector<int>> transpose(std::vector<int> shape, std::vector<int> strides) {
    std::swap(shape[shape.size()-1], shape[shape.size()-2]);
    std::swap(strides[strides.size()-1], strides[strides.size()-2]);
    return {shape, strides};
}

inline int idx_nd_to_1d(const std::vector<int>& idxs, const std::vector<int>& strides) {
    int idx_1d = 0;
    for (int d = 0; d < (int)idxs.size(); d++)
        idx_1d += idxs[d] * strides[d];
    return idx_1d;
}

// converts a flat index to an n-dimensional index given shape
inline std::vector<int> idx_1d_to_nd(int idx_1d, std::vector<int> shape) {
    std::vector<int> idxs(shape.size());
    for (int dim = shape.size() - 1; dim >= 0; dim--) {
        idxs[dim] = idx_1d % shape[dim];
        idx_1d /= shape[dim];
    }
    return idxs;
}

inline std::vector<int> squeeze_shape(std::vector<int> shape, int dim) {
    if (shape[dim] != 1) throw std::logic_error("Can only squeeze a dimension of size 1.");
    shape.erase(shape.begin() + dim);
    return shape;
}

inline std::vector<int> unsqueeze_shape(std::vector<int> shape, int dim) {
    shape.insert(shape.begin() + dim, 1);
    return shape;
}

inline std::vector<int> flatten(std::vector<int> shape, int d1 = 0, int d2 = -1) {
    int ndim = shape.size();
    if (d2 < 0) d2 = ndim + d2;  // handle negative indexing
    std::vector<int> out_shape;
    int flat_size = 1;
    for (int i = 0; i < ndim; i++) {
        if (i < d1 || i > d2) {
            out_shape.push_back(shape[i]);
        } else {
            flat_size *= shape[i];
        }
    }
    out_shape.insert(out_shape.begin() + d1, flat_size);
    return out_shape;
}

inline void check_matmul_shapes(const std::vector<int>& s1, const std::vector<int>& s2) {
    if (!(s1.size() >= 2 && s2.size() >= 2))
        throw std::logic_error("Tensors rank should be at least 2.");
    if (s1.back() != s2[s2.size() - 2])
        throw std::logic_error("k dimensions don't match.");
    for (int i = 0; i < (int)s1.size() - 2; i++)
        if (s1[i] != s2[i]) throw std::logic_error("Batch dimensions should match.");
}

inline bool is_scalar(const std::vector<int>& shape) {
    return shape.size() == 1 && shape[0] == 1;
}

inline bool same_shape(const std::vector<int>& a, const std::vector<int>& b) {
    return a == b;
}

inline std::pair<std::vector<int>, int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
    int ndim = std::max(a.size(), b.size());
    std::vector<int> result(ndim);
    for (int i = 0; i < ndim; i++) {
        int a_dim = i < a.size() ? a[a.size() - 1 - i] : 1;  // pad with 1s from left
        int b_dim = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw std::logic_error("shapes not broadcastable");
        result[ndim - 1 - i] = std::max(a_dim, b_dim);
    }
    return {result, ndim};
}

struct MatmulDims {
    int M, K, N, batch;
    std::vector<int> res_shape;
};

inline MatmulDims get_matmul_dims(const std::vector<int>& a_shape, const std::vector<int>& b_shape) {
    int ndim = a_shape.size();
    int M = a_shape[ndim - 2];
    int K = a_shape[ndim - 1];
    int N = b_shape[ndim - 1];
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) batch *= a_shape[i];

    std::vector<int> res_shape(a_shape.begin(), a_shape.end() - 2);
    res_shape.push_back(M);
    res_shape.push_back(N);

    return {M, K, N, batch, res_shape};
}


}


#ifdef __CUDACC__
  #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif

namespace operators {
struct Add { HOST_DEVICE float operator()(float a, float b) { return a + b; } };
struct Mul { HOST_DEVICE float operator()(float a, float b) { return a * b; } };
struct Div { HOST_DEVICE float operator()(float a, float b) { return a / b; } };
struct Sub { HOST_DEVICE float operator()(float a, float b) { return a - b; } };
}
