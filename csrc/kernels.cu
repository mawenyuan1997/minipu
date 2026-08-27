#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void custom_add_kernel(
    const float* x,
    const float* y,
    float* output,
    int64_t numel,
    float alpha
) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (index < numel) {
        output[index] = x[index] + alpha * y[index];
    }
}

torch::Tensor my_gpu_add(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const c10::Scalar& alpha
) {
    TORCH_CHECK(
        x.device().type() == c10::DeviceType::PrivateUse1,
        "x must be a MiniPU tensor"
    );

    TORCH_CHECK(
        y.device().type() == c10::DeviceType::PrivateUse1,
        "y must be a MiniPU tensor"
    );

    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32,
        "x must have dtype torch.float32"
    );

    TORCH_CHECK(
        y.scalar_type() == torch::kFloat32,
        "y must have dtype torch.float32"
    );

    TORCH_CHECK(
        x.sizes() == y.sizes(),
        "x and y must have the same shape"
    );

    TORCH_CHECK(
        x.is_contiguous(),
        "x must be contiguous"
    );

    TORCH_CHECK(
        y.is_contiguous(),
        "y must be contiguous"
    );

    auto output = torch::empty_like(x);

    const int64_t numel = x.numel();

    if (numel == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int64_t blocks = (numel + threads - 1) / threads;
    const float alpha_value = alpha.to<float>();

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream().stream();

    custom_add_kernel<<<
        static_cast<unsigned int>(blocks),
        threads,
        0,
        stream
    >>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        alpha_value
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", TORCH_FN(my_gpu_add));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}