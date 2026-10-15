// The ATen, c10 and pybind11 surface python-0.6's qmm and experts extensions use, standing in for torch so their own C++
// wrappers run on a CPU (qualification_765330ce's tools/zig/exl3_launches design): a tensor is an address, a dtype, sizes and strides;
// TORCH_CHECK throws; PYBIND11_MODULE registers the module's functions with glm_mm_launches.cpp, which also stands in
// for the CUDA runtime and prints each launch.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace c10 {
enum class ScalarType : int8_t { Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16 };
template <class T>
using optional = std::optional<T>;
struct Device {
    int16_t index = 0;
    bool operator==(const Device& o) const { return index == o.index; }
};
}  // namespace c10

namespace at {
using c10::ScalarType;
constexpr ScalarType kByte = ScalarType::Byte, kChar = ScalarType::Char, kShort = ScalarType::Short,
                     kInt = ScalarType::Int, kLong = ScalarType::Long, kHalf = ScalarType::Half,
                     kFloat = ScalarType::Float, kDouble = ScalarType::Double, kBFloat16 = ScalarType::BFloat16;

struct TensorOptions {
    ScalarType st = ScalarType::Float;
    TensorOptions dtype(ScalarType s) const { return {s}; }
};

class Tensor {
   public:
    uintptr_t addr = 0;
    ScalarType st = ScalarType::Float;
    std::vector<int64_t> sz, sd;

    int64_t dim() const { return (int64_t)sz.size(); }
    int64_t size(int64_t d) const { return sz.at(d < 0 ? d + dim() : d); }
    int64_t stride(int64_t d) const { return sd.at(d < 0 ? d + dim() : d); }
    const std::vector<int64_t>& sizes() const { return sz; }
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : sz) n *= s;
        return n;
    }
    int64_t element_size() const {
        switch (st) {
            case ScalarType::Byte: case ScalarType::Char: return 1;
            case ScalarType::Short: case ScalarType::Half: case ScalarType::BFloat16: return 2;
            case ScalarType::Int: case ScalarType::Float: return 4;
            default: return 8;
        }
    }
    bool is_cuda() const { return true; }
    bool defined() const { return addr != 0 || !sz.empty(); }  // a default-constructed tensor is undefined
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
    TensorOptions options() const { return {st}; }
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

template <class T>
T* Tensor::data_ptr() const {
    if (st != scalar_of<T>()) throw std::runtime_error("data_ptr<T>() of a tensor of another dtype");
    return reinterpret_cast<T*>(addr);
}

// only qmm()'s split-K scratch allocates, and GLM's grouped path never reaches it
inline Tensor empty(std::initializer_list<int64_t>, TensorOptions) {
    throw std::runtime_error("at::empty: an allocation inside the wrapper (not on GLM's path)");
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
namespace at::cuda {
using CUDAGuard = c10::cuda::CUDAGuard;
}

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
    enum Kind { T, I, F, B, N, L } kind = N;
    at::Tensor t;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::vector<Arg> list;
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
struct Conv<bool> {
    static bool get(const Arg& a) {
        if (a.kind != Arg::B) throw std::runtime_error("expected a bool");
        return a.b;
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
template <class E>
struct Conv<std::vector<E>> {
    static std::vector<E> get(const Arg& a) {
        if (a.kind != Arg::L) throw std::runtime_error("expected a list");
        std::vector<E> v;
        for (const auto& x : a.list) v.push_back(Conv<E>::get(x));
        return v;
    }
};
using Fn = std::function<void(const std::vector<Arg>&)>;
struct Module {
    std::map<std::string, Fn> fns;
    template <class R, class... A>
    void def(const char* name, R (*f)(A...), const char* = nullptr) {
        fns[name] = [f](const std::vector<Arg>& v) {
            if (v.size() != sizeof...(A)) throw std::runtime_error("wrong number of arguments");
            call(f, v, std::index_sequence_for<A...>{});
        };
    }
    // each argument converted once and passed as an lvalue, so `at::Tensor&` parameters bind as pybind11 binds them
    template <class R, class... A, size_t... I>
    static void call(R (*f)(A...), const std::vector<Arg>& v, std::index_sequence<I...>) {
        std::tuple<std::decay_t<A>...> args{Conv<std::decay_t<A>>::get(v[I])...};
        f(std::get<I>(args)...);
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
