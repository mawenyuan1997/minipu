#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace {

void cuda_deleter(void* ptr) {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
    }
}

class MyGpuAllocator final : public c10::Allocator {
public:
    c10::DataPtr allocate(std::size_t nbytes) override {
        int current_device = 0;

        cudaError_t status = cudaGetDevice(&current_device);
        TORCH_CHECK(
            status == cudaSuccess,
            "cudaGetDevice failed: ",
            cudaGetErrorString(status)
        );

        if (nbytes == 0) {
            return c10::DataPtr(
                nullptr,
                c10::Device(
                    c10::DeviceType::PrivateUse1,
                    static_cast<c10::DeviceIndex>(current_device)
                )
            );
        }

        void* ptr = nullptr;
        status = cudaMalloc(&ptr, nbytes);

        TORCH_CHECK(
            status == cudaSuccess,
            "cudaMalloc failed for ",
            nbytes,
            " bytes: ",
            cudaGetErrorString(status)
        );

        return c10::DataPtr(
            ptr,
            ptr,
            &cuda_deleter,
            c10::Device(
                c10::DeviceType::PrivateUse1,
                static_cast<c10::DeviceIndex>(current_device)
            )
        );
    }

    void copy_data(
        void* destination,
        const void* source,
        std::size_t count
    ) const override {
        if (count == 0) {
            return;
        }

        const cudaError_t status = cudaMemcpy(
            destination,
            source,
            count,
            cudaMemcpyDeviceToDevice
        );

        TORCH_CHECK(
            status == cudaSuccess,
            "cudaMemcpyDeviceToDevice failed for ",
            count,
            " bytes: ",
            cudaGetErrorString(status)
        );
    }

    c10::DeleterFnPtr raw_deleter() const override {
        return &cuda_deleter;
    }
};

MyGpuAllocator my_allocator_instance;

} // namespace

REGISTER_ALLOCATOR(
    c10::DeviceType::PrivateUse1,
    &my_allocator_instance
);