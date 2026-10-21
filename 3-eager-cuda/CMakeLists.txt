cmake_minimum_required(VERSION 3.18)
project(bobrito LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

find_package(CUDAToolkit REQUIRED)

pybind11_add_module(bobrito_cpp
    src/bindings.cpp
    src/tensor.cpp
    src/autograd.cpp
    src/ops.cpp
    src/cuda_ops.cu
)

target_link_libraries(bobrito_cpp PRIVATE
    CUDA::cublas
    CUDA::cudart
    $ENV{CONDA_PREFIX}/lib/libopenblas.so
    $ENV{CONDA_PREFIX}/lib/libcutensor.so
)

target_compile_options(bobrito_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O2
        --use_fast_math
        -gencode arch=compute_120,code=sm_120
    >
)

install(TARGETS bobrito_cpp DESTINATION bobrito)
