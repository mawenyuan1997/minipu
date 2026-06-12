#include <ATen/native/CPUFallback.h>
#include <torch/extension.h>
#include <torch/library.h>


namespace {

void minipu_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

}  // namespace


TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&minipu_cpu_fallback>());
}


TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("is_registered", []() { return true; });
}
