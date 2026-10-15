// The ATen, c10 and pybind11 surface python-0.6's EXL3 extensions use, standing in for torch so their own C++ wrappers
// run on a CPU: a tensor is an address, a dtype, sizes and strides; TORCH_CHECK throws; PYBIND11_MODULE registers the
// module's functions with exl3_launches.cpp, which also stands in for the CUDA runtime and records each launch.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace c10 {
enum class ScalarType : int8_t { Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16 };
template <class T>
using optional = std::optional<T>;
struct Device {
    int16_t index = 0;
};
}  // namespace c10

namespace at {
using c10::ScalarType;
constexpr ScalarType kByte = ScalarType::Byte, kChar = ScalarType::Char, kShort = ScalarType::Short,
                     kInt = ScalarType::Int, kLong = ScalarType::Long, kHalf = ScalarType::Half,
                     kFloat = ScalarType::Float, kDouble = ScalarType::Double, kBFloat16 = ScalarType::BFloat16;

class Tensor {
   public:
    uintptr_t addr = 0;
    ScalarType st = ScalarType::Float;
    std::vector<int64_t> sz, sd;
    bool cuda = true;

    int64_t dim() const { return (int64_t)sz.size(); }
    int64_t size(int64_t d) const { return sz.at(d < 0 ? d + dim() : d); }
    int64_t stride(int64_t d) const { return sd.at(d < 0 ? d + dim() : d); }
    const std::vector<int64_t>& sizes() const { return sz; }
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : sz) n *= s;
        return n;
    }
    bool is_cuda() const { return cuda; }
    bool is_contiguous() const {  // torch's rule: dims of size 1 may have any stride
        int64_t want = 1;
        for (int64_t i = dim() - 1; i >= 0; --i) {
            if (sz[i] != 1 && sd[i] != want) return false;
            want *= sz[i];
        }
        return true;
    }
    ScalarType scalar_type() const { return st; }
    c10::Device device() const { return {}; }
    void* data_ptr() const { return reinterpret_cast<void*>(addr); }
    template <class T>
    T* data_ptr() const;
};

template <class T>
constexpr ScalarType scalar_of();
template <>
constexpr ScalarType scalar_of<int>() { return ScalarType::Int; }
template <>
constexpr ScalarType scalar_of<int64_t>() { return ScalarType::Long; }
template <>
constexpr ScalarType scalar_of<float>() { return ScalarType::Float; }
template <>
constexpr ScalarType scalar_of<int16_t>() { return ScalarType::Short; }

template <class T>
T* Tensor::data_ptr() const {
    if (st != scalar_of<T>()) throw std::runtime_error("data_ptr<T>() of a tensor of another dtype");
    return reinterpret_cast<T*>(addr);
}

namespace cuda {
struct CUDAStream {
    cudaStream_t s;
    operator cudaStream_t() const { return s; }
    cudaStream_t stream() const { return s; }
};
CUDAStream getCurrentCUDAStream(int device = -1);
cudaDeviceProp* getCurrentDeviceProperties();
}  // namespace cuda
}  // namespace at

namespace c10::cuda {
struct CUDAGuard {
    explicit CUDAGuard(c10::Device) {}
};
}  // namespace c10::cuda

template <class... A>
std::string tf_str(const A&... a) {
    std::ostringstream o;
    (o << ... << a);
    return o.str();
}
#define TORCH_CHECK(cond, ...) \
    do {                       \
        if (!(cond)) throw std::runtime_error(tf_str("" __VA_OPT__(, ) __VA_ARGS__)); \
    } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() \
    do {                               \
    } while (0)
#define C10_CUDA_CHECK(x) \
    do {                  \
        (void)(x);        \
    } while (0)

namespace tf {
struct Arg {
    enum Kind { T, I, F, N } kind = N;
    at::Tensor t;
    int64_t i = 0;
    double f = 0;
};
template <class T>
struct Conv;
template <>
struct Conv<at::Tensor> {
    static at::Tensor get(const Arg& a) {
        if (a.kind != Arg::T) throw std::runtime_error("expected a tensor");
        return a.t;
    }
};
template <>
struct Conv<int64_t> {
    static int64_t get(const Arg& a) {
        if (a.kind != Arg::I) throw std::runtime_error("expected an int");
        return a.i;
    }
};
template <>
struct Conv<double> {
    static double get(const Arg& a) {
        if (a.kind == Arg::F) return a.f;
        if (a.kind == Arg::I) return (double)a.i;
        throw std::runtime_error("expected a float");
    }
};
template <>
struct Conv<c10::optional<at::Tensor>> {
    static c10::optional<at::Tensor> get(const Arg& a) {
        if (a.kind == Arg::N) return std::nullopt;
        return Conv<at::Tensor>::get(a);
    }
};
using Fn = std::function<void(const std::vector<Arg>&)>;
struct Module {
    std::map<std::string, Fn> fns;
    template <class R, class... A>
    void def(const char* name, R (*f)(A...)) {
        fns[name] = [f](const std::vector<Arg>& v) {
            if (v.size() != sizeof...(A)) throw std::runtime_error("wrong number of arguments");
            call(f, v, std::index_sequence_for<A...>{});
        };
    }
    template <class R, class... A, size_t... I>
    static void call(R (*f)(A...), const std::vector<Arg>& v, std::index_sequence<I...>) {
        f(Conv<std::decay_t<A>>::get(v[I])...);
    }
};
void add_module(const char* name, void (*init)(Module&));
struct Reg {
    Reg(const char* name, void (*init)(Module&)) { add_module(name, init); }
};
}  // namespace tf

#define TF_STR2(x) #x
#define TF_STR(x) TF_STR2(x)
#define PYBIND11_MODULE(name, m)                               \
    static void tf_module_init(tf::Module& m);                 \
    static tf::Reg tf_module_reg(TF_STR(name), tf_module_init); \
    static void tf_module_init(tf::Module& m)
