#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cuda_runtime.h>

class MyDeviceGuardImpl final
    : public c10::impl::DeviceGuardImplInterface {
public:
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device device) const override {
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device.index());

        return c10::Device(
            c10::DeviceType::PrivateUse1,
            static_cast<c10::DeviceIndex>(previous_device)
        );
    }

    c10::Device getDevice() const override {
        int current_device = 0;
        cudaGetDevice(&current_device);

        return c10::Device(
            c10::DeviceType::PrivateUse1,
            static_cast<c10::DeviceIndex>(current_device)
        );
    }

    void setDevice(c10::Device device) const override {
        cudaSetDevice(device.index());
    }

    void uncheckedSetDevice(
        c10::Device device
    ) const noexcept override {
        // 该函数是 noexcept，因此不能在这里抛出异常。
        (void)cudaSetDevice(device.index());
    }

    c10::Stream getStream(c10::Device device) const override {
        return c10::Stream(
            c10::Stream::DEFAULT,
            device
        );
    }

    c10::Stream exchangeStream(c10::Stream stream) const override {
        // 当前 demo 还没有实现 PrivateUse1 stream 状态管理。
        return stream;
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);

        if (status != cudaSuccess) {
            // 接口要求 noexcept；发生驱动错误时返回 0。
            return 0;
        }

        return static_cast<c10::DeviceIndex>(count);
    }

    void synchronizeDevice(
        const c10::DeviceIndex device_index
    ) const override {
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        cudaSetDevice(device_index);
        cudaDeviceSynchronize();
        cudaSetDevice(previous_device);
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, MyDeviceGuardImpl);