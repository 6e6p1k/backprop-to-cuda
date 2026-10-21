#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for automatic vector conversion
#include "./tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<float>, std::vector<int>, bool>(),
             py::arg("values"), py::arg("shape"), py::arg("requires_grad") = false)
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("matmul", &Tensor::matmul)
        .def("matmul_blas", &Tensor::matmul_blas)
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
        .def("update", &Tensor::update)
        .def("backward", &Tensor::backward)
        .def("print", &Tensor::print)
        .def("eq", &Tensor::eq)
        .def_property_readonly("grad", [](const Tensor& t) { return t.td->grad; })
        .def_property_readonly("shape", [](const Tensor& t) { return t.td->shape; })
        .def_property_readonly("values", [](const Tensor& t) { return t.td->values; })
        .def_readonly("requires_grad", &Tensor::requires_grad);

}