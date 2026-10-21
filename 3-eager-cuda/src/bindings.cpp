#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(bobrito_cpp, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>, std::string, bool>(),
            py::arg("values"), 
            py::arg("shape"), 
            py::arg("device") = "cpu", 
            py::arg("requires_grad") = false
        )
        .def (py::init<std::vector<int>, float, std::string, bool>(),
            py::arg("shape"), 
            py::arg("fill_value"), 
            py::arg("device") = "cpu", 
            py::arg("requires_grad") = false
        )
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("matmul", &Tensor::matmul)
        .def("matmul_fast", &Tensor::matmul_fast)
        .def("matmul_blas", &Tensor::matmul_blas)
        .def("matmul_cuda", &Tensor::matmul_cuda)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("max", &Tensor::max)
        .def("argmax", &Tensor::argmax)
        .def("log", &Tensor::log)
        .def("exp", &Tensor::exp)
        .def("pow", &Tensor::pow)
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)
        .def("squeeze", &Tensor::squeeze)
        .def("unsqueeze", &Tensor::unsqueeze)
        .def("flatten", &Tensor::flatten)
        .def("reset_grad", &Tensor::reset_grad)
        .def("sgd_update", &Tensor::sgd_update)
        .def("backward", &Tensor::backward)
        .def("eq", &Tensor::eq)
        .def("to_cuda", &Tensor::to_cuda)
        .def("to_cpu",  &Tensor::to_cpu)
        .def("_out_full_values", &Tensor::to_host_full)
        .def("_out_truncated_values", &Tensor::to_host_truncated)
        .def("_out_size", &Tensor::numel)
        .def_property_readonly("grad", [](const Tensor& t) { return t.td->grad; })
        .def_property_readonly("shape", [](const Tensor& t) { return t.td->shape; })
        .def_property_readonly("device", [](const Tensor& t) { return deviceToString(t.td->device); })

        .def_readonly("requires_grad", &Tensor::requires_grad);

}