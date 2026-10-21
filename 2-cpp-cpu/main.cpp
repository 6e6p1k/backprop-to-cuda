// imports
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <set>
#include <functional>
#include <memory>

#include "./tensor.h"

// helpers
static void check(const std::string& name,
                  const std::vector<float>& got_data,  const std::vector<float>& exp_data,
                  const std::vector<int>&   got_shape, const std::vector<int>&   exp_shape,
                  float tol = 1e-4f)
{
    bool ok = true;
    if (got_shape != exp_shape) { ok = false; }
    if (got_data.size() != exp_data.size()) { ok = false; }
    else {
        for (size_t i = 0; i < exp_data.size(); i++)
            if (std::fabs(got_data[i] - exp_data[i]) > tol) { ok = false; break; }
    }
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) {
        std::cout << "  got  shape:"; for (int s : got_shape)  std::cout << " " << s; std::cout << "\n";
        std::cout << "  exp  shape:"; for (int s : exp_shape)  std::cout << " " << s; std::cout << "\n";
        std::cout << "  got  data: "; for (float v : got_data)  std::cout << v << " "; std::cout << "\n";
        std::cout << "  exp  data: "; for (float v : exp_data)  std::cout << v << " "; std::cout << "\n";
    }
}

static void check_grad(const std::string& name, bool has_grad_fn) {
    std::cout << (has_grad_fn ? "[PASS] " : "[FAIL] ") << name << "\n";
}

static void check_vals(const std::string& name,
                       const std::vector<float>& got, const std::vector<float>& exp,
                       float tol = 1e-4f)
{
    bool ok = (got.size() == exp.size());
    if (ok) for (size_t i = 0; i < exp.size(); i++)
        if (std::fabs(got[i] - exp[i]) > tol) { ok = false; break; }
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) {
        std::cout << "  got: "; for (float v : got) std::cout << v << " "; std::cout << "\n";
        std::cout << "  exp: "; for (float v : exp) std::cout << v << " "; std::cout << "\n";
    }
}

int main(int argc, const char * argv[]) {

    // autograd backward tests
    // add: da = db = 1
    {
        Tensor a({1,2,3}, {3}, true);
        Tensor b({4,5,6}, {3}, true);
        a.add(b).backward();
        check_vals("add backward da", a.td->grad, {1,1,1});
        check_vals("add backward db", b.td->grad, {1,1,1});
    }

    // mul: da = b, db = a
    {
        Tensor a({1,2,3}, {3}, true);
        Tensor b({4,5,6}, {3}, true);
        a.mul(b).backward();
        check_vals("mul backward da", a.td->grad, {4,5,6});
        check_vals("mul backward db", b.td->grad, {1,2,3});
    }

    // log: da = 1/a
    {
        Tensor a({1,2,4}, {3}, true);
        a.log().backward();
        check_vals("log backward", a.td->grad, {1.0f, 0.5f, 0.25f});
    }

    // exp: da = exp(a)
    {
        Tensor a({0,1,2}, {3}, true);
        a.exp().backward();
        check_vals("exp backward", a.td->grad, {1.0f, std::exp(1.0f), std::exp(2.0f)});
    }

    // pow: da = p * a^(p-1)
    {
        Tensor a({1,2,3}, {3}, true);
        a.pow(2.0f).backward();
        check_vals("pow backward", a.td->grad, {2.0f, 4.0f, 6.0f});
    }

    // sum: grad broadcast back to all inputs
    {
        Tensor a({1,2,3,4}, {2,2}, true);
        a.sum(0, false).backward();
        check_vals("sum backward", a.td->grad, {1,1,1,1});
    }

    // mean: grad / N
    {
        Tensor a({1,2,3,4}, {2,2}, true);
        a.mean(0, false).backward();  // N=2
        check_vals("mean backward", a.td->grad, {0.5f, 0.5f, 0.5f, 0.5f});
    }

    // chain: (a * b).sum -> da = b
    {
        Tensor a({2,3}, {2}, true);
        Tensor b({4,5}, {2}, false);
        a.mul(b).sum(0, false).backward();
        check_vals("chain mul+sum backward da", a.td->grad, {4,5});
    }

    // matmul with identity: grad_A = ones, grad_B = A^T @ ones
    {
        Tensor A({1,2,3,4}, {2,2}, true);
        Tensor B({1,0,0,1}, {2,2}, true);
        A.matmul(B).backward();
        check_vals("matmul backward dA", A.td->grad, {1,1,1,1});
        check_vals("matmul backward dB", B.td->grad, {4,4,6,6});
    }

    return EXIT_SUCCESS;
}
