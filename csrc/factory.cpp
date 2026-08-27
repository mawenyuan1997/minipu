#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <torch/library.h>

#include <optional>

namespace {

void check_factory_options(
    const std::optional<c10::Layout>& layout,
    const std::optional<c10::Device>& device,
    const std::optional<bool>& pin_memory
) {
    TORCH_CHECK(
        !layout.has_value() ||
            layout.value() == c10::Layout::Strided,
        "MiniPU currently supports strided layout only"
    );

    TORCH_CHECK(
        !device.has_value() ||
            device->type() == c10::DeviceType::PrivateUse1,
        "MiniPU factory received an invalid device: ",
        device.has_value() ? device->str() : "None"
    );

    TORCH_CHECK(
        !pin_memory.value_or(false),
        "MiniPU does not support pinned device memory"
    );
}

c10::ScalarType resolve_dtype(
    const std::optional<c10::ScalarType>& dtype
) {
    return dtype.value_or(
        c10::get_default_dtype_as_scalartype()
    );
}

at::Tensor minipu_empty(
    c10::SymIntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format
) {
    check_factory_options(layout, device, pin_memory);

    c10::Allocator* allocator = c10::GetAllocator(
        c10::DeviceType::PrivateUse1
    );

    const c10::DispatchKeySet dispatch_keys(
        c10::DispatchKey::PrivateUse1
    );

    return at::Tensor(
        at::detail::empty_generic_symint(
            size,
            allocator,
            dispatch_keys,
            resolve_dtype(dtype),
            memory_format
        )
    );
}

at::Tensor minipu_empty_strided(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory
) {
    check_factory_options(layout, device, pin_memory);

    TORCH_CHECK(
        size.size() == stride.size(),
        "size and stride must have the same length"
    );

    c10::Allocator* allocator = c10::GetAllocator(
        c10::DeviceType::PrivateUse1
    );

    const c10::DispatchKeySet dispatch_keys(
        c10::DispatchKey::PrivateUse1
    );

    return at::Tensor(
        at::detail::empty_strided_symint_generic(
            size,
            stride,
            allocator,
            dispatch_keys,
            resolve_dtype(dtype)
        )
    );
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl(
        "empty.memory_format",
        TORCH_FN(minipu_empty)
    );

    m.impl(
        "empty_strided",
        TORCH_FN(minipu_empty_strided)
    );
}