// python-0.6's EXL3 extensions' own C++ wrappers (experts.cpp/.cu, linear.cpp/.cu, GLM's exl3.cpp/.cu, compiled from
// the python-0.6 tree as they are) run on a CPU: the calls python-0.6's Python wrappers made (recorded by
// tools/zig/exl3_fixtures.py) come in, each launch the wrappers make goes out. This file stands in for the CUDA runtime
// those translation units link against: kernel registration, <<<>>> (push/pop configuration, cudaLaunchKernel) and the
// function attributes. Arguments are copied at the sizes the cubins' parameter tables give (params.txt).
//
// in (stdin):  CASE id | OPTIN bytes | CALL module fn nargs, then per argument: T addr dtype ndim sizes.. strides.. |
//              I int | F double-bits | N
// out (stdout): CASE id | LAUNCH call name gx gy gz bx by bz smem n hex.. | GETATTR call name |
//              ATTR call name attr value | ERROR call message
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "tf_fake.h"

namespace {
std::map<std::string, tf::Module>& modules() {
    static std::map<std::string, tf::Module> m;
    return m;
}
std::map<const void*, std::string>& names() {
    static std::map<const void*, std::string> n;
    return n;
}
struct Kernel {
    std::vector<int> sizes;
    int shared = 0;
};
std::map<std::string, Kernel> table;           // "module key" -> parameter sizes, static shared bytes
std::map<const void*, int> max_dynamic;        // cudaFuncAttributeMaxDynamicSharedMemorySize as set so far
std::string current_module;
int call_no = -1;
int optin = 0;
struct Config {
    dim3 g, b;
    size_t smem;
    cudaStream_t s;
};
std::vector<Config> configs;

// The mangled name past its namespace (_ZN<n><n chars>): python-0.6's anonymous namespaces, our named ones.
std::string key(const std::string& name) {
    if (name.rfind("_ZN", 0) != 0) return name;
    size_t i = 3, n = 0;
    while (i < name.size() && isdigit((unsigned char)name[i])) n = n * 10 + (name[i++] - '0');
    return name.substr(i + n);
}

const Kernel& kernel(const void* f) {
    auto it = names().find(f);
    if (it == names().end()) throw std::runtime_error("launch of an unregistered kernel");
    auto k = table.find(current_module + " " + key(it->second));
    if (k == table.end()) throw std::runtime_error("no parameter table for " + it->second);
    return k->second;
}
}  // namespace

namespace tf {
void add_module(const char* name, void (*init)(Module&)) { init(modules()[name]); }
}  // namespace tf

namespace at::cuda {
CUDAStream getCurrentCUDAStream(int) { return {reinterpret_cast<cudaStream_t>(0x57)}; }
cudaDeviceProp* getCurrentDeviceProperties() {
    static cudaDeviceProp p;
    p.sharedMemPerBlockOptin = (size_t)optin;
    return &p;
}
}  // namespace at::cuda

extern "C" {
void** __cudaRegisterFatBinary(void*) {
    static void* handle = nullptr;
    return &handle;
}
void __cudaRegisterFatBinaryEnd(void**) {}
void __cudaUnregisterFatBinary(void**) {}
void __cudaRegisterFunction(void**, const char* host, char*, const char* device, int, uint3*, uint3*, dim3*, dim3*,
                            int*) {
    names()[host] = device;
}
void __cudaRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
unsigned __cudaPushCallConfiguration(dim3 g, dim3 b, size_t smem, struct CUstream_st* s) {
    configs.push_back({g, b, smem, s});
    return 0;
}
cudaError_t __cudaPopCallConfiguration(dim3* g, dim3* b, size_t* smem, void* s) {
    Config c = configs.back();
    configs.pop_back();
    *g = c.g, *b = c.b, *smem = c.smem;
    *reinterpret_cast<cudaStream_t*>(s) = c.s;
    return cudaSuccess;
}
cudaError_t cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t smem, cudaStream_t) {
    const Kernel& k = kernel(f);
    printf("LAUNCH %d %s %u %u %u %u %u %u %zu %zu", call_no, names()[f].c_str(), g.x, g.y, g.z, b.x, b.y, b.z, smem,
           k.sizes.size());
    for (size_t i = 0; i < k.sizes.size(); ++i) {
        printf(" ");
        const unsigned char* p = static_cast<const unsigned char*>(args[i]);
        for (int j = 0; j < k.sizes[i]; ++j) printf("%02x", p[j]);
    }
    printf("\n");
    return cudaSuccess;
}
// CUDA 13's stubs: the kernel's handle, then the launch through it (crt/device_functions.h, crt/host_runtime.h)
cudaError_t __cudaGetKernel(cudaKernel_t* handle, const void* f) {
    *handle = (cudaKernel_t)f;
    return cudaSuccess;
}
cudaError_t __cudaLaunchKernel(cudaKernel_t k, dim3 g, dim3 b, void** args, size_t smem, cudaStream_t s) {
    return cudaLaunchKernel((const void*)k, g, b, args, smem, s);
}
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void* f) {
    const Kernel& k = kernel(f);
    memset(a, 0, sizeof(*a));
    a->sharedSizeBytes = (size_t)k.shared;
    auto it = max_dynamic.find(f);
    a->maxDynamicSharedSizeBytes = it != max_dynamic.end() ? it->second : 48 * 1024 - k.shared;
    printf("GETATTR %d %s\n", call_no, names()[f].c_str());
    return cudaSuccess;
}
cudaError_t cudaFuncSetAttribute(const void* f, cudaFuncAttribute attr, int value) {
    kernel(f);
    if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) max_dynamic[f] = value;
    printf("ATTR %d %s %d %d\n", call_no, names()[f].c_str(), (int)attr, value);
    return cudaSuccess;
}
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: exl3_launches params.txt < calls.txt\n");
        return 2;
    }
    std::ifstream pt(argv[1]);
    std::string mod, k;
    int n;
    while (pt >> mod >> k >> n) {
        Kernel e;
        e.sizes.resize(n);
        for (int& s : e.sizes) pt >> s;
        pt >> e.shared;
        table[mod + " " + k] = e;
    }
    std::string word;
    while (std::cin >> word) {
        if (word == "CASE") {
            std::string id;
            std::cin >> id;
            max_dynamic.clear();  // each case a fresh process
            call_no = -1;
            printf("CASE %s\n", id.c_str());
        } else if (word == "OPTIN") {
            std::cin >> optin;
        } else if (word == "CALL") {
            std::string fn;
            size_t nargs;
            std::cin >> current_module >> fn >> nargs;
            ++call_no;
            std::vector<tf::Arg> args(nargs);
            for (auto& a : args) {
                std::string t;
                std::cin >> t;
                if (t == "T") {
                    a.kind = tf::Arg::T;
                    uint64_t addr;
                    int dtype, ndim;
                    std::cin >> std::hex >> addr >> std::dec >> dtype >> ndim;
                    a.t.addr = addr;
                    a.t.st = (c10::ScalarType)dtype;
                    a.t.sz.resize(ndim);
                    a.t.sd.resize(ndim);
                    for (auto& s : a.t.sz) std::cin >> s;
                    for (auto& s : a.t.sd) std::cin >> s;
                } else if (t == "I") {
                    a.kind = tf::Arg::I;
                    std::cin >> a.i;
                } else if (t == "F") {
                    a.kind = tf::Arg::F;
                    uint64_t bits;
                    std::cin >> std::hex >> bits >> std::dec;
                    memcpy(&a.f, &bits, 8);
                } else {
                    a.kind = tf::Arg::N;
                }
            }
            try {
                auto m = modules().find(current_module);
                if (m == modules().end()) throw std::runtime_error("no module " + current_module);
                auto f = m->second.fns.find(fn);
                if (f == m->second.fns.end()) throw std::runtime_error("no function " + fn);
                f->second(args);
            } catch (const std::exception& e) {
                printf("ERROR %d %s\n", call_no, e.what());
            }
        }
    }
    return 0;
}
