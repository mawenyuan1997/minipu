cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_gpu_backend LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

add_library(my_gpu_backend SHARED 
    csrc/allocator.cpp 
    csrc/runtime.cpp 
    csrc/kernels.cu
)

target_link_libraries(my_gpu_backend PRIVATE ${TORCH_LIBRARIES})
target_compile_options(my_gpu_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>>:-O3)
