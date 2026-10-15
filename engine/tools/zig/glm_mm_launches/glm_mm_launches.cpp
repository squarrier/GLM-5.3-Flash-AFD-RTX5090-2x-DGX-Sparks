// python-0.6's qmm and experts extensions' own C++ wrappers (qmm.cpp + qmm*.cu, experts.cpp + experts*.cu, compiled from
// the python-0.6 tree as they are) run on a CPU: the calls python-0.6's Python wrappers made (recorded by
// tools/zig/glm_mm_fixtures.py) come in, every launch the wrappers make goes out (qualification_765330ce's exl3_launches design). This file
// stands in for the CUDA runtime those translation units link against: kernel registration, <<<>>>, cudaLaunchKernelEx
// with its cluster and PDL attributes, function attributes and occupancy. Arguments are copied at the sizes the cubins'
// parameter tables give (params.txt: key n size.. per kernel).
//
// in (stdin):  CASE id | CHIP major minor sms optin | OCC blocks | CALL module fn nargs, then per argument:
//              T addr dtype ndim sizes.. strides.. | I int | F double-bits | B 0|1 | N | L n args..
// out (stdout): CASE id | LAUNCH call name gx gy gz bx by bz smem cx cy cz pdl n hex.. | ATTR call name attr value |
//              OCCQ call name threads smem | ERROR call message
#include <cuda_runtime.h>

#include <cctype>
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
std::map<std::string, std::vector<int>> table;  // kernel key -> parameter sizes
int call_no = -1;
int occupancy = 1;
cudaDeviceProp props;
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

const std::string& name_of(const void* f) {
    auto it = names().find(f);
    if (it == names().end()) throw std::runtime_error("launch of an unregistered kernel");
    return it->second;
}

void print_launch(const void* f, dim3 g, dim3 b, size_t smem, dim3 c, int pdl, void** args) {
    const std::string& name = name_of(f);
    auto k = table.find(key(name));
    if (k == table.end()) throw std::runtime_error("no parameter table for " + name);
    printf("LAUNCH %d %s %u %u %u %u %u %u %zu %u %u %u %d %zu", call_no, name.c_str(), g.x, g.y, g.z, b.x, b.y, b.z,
           smem, c.x, c.y, c.z, pdl, k->second.size());
    for (size_t i = 0; i < k->second.size(); ++i) {
        printf(" ");
        const unsigned char* p = static_cast<const unsigned char*>(args[i]);
        for (int j = 0; j < k->second[i]; ++j) printf("%02x", p[j]);
    }
    printf("\n");
}

tf::Arg read_arg() {
    tf::Arg a;
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
    } else if (t == "B") {
        a.kind = tf::Arg::B;
        int v;
        std::cin >> v;
        a.b = v != 0;
    } else if (t == "L") {
        a.kind = tf::Arg::L;
        size_t n;
        std::cin >> n;
        for (size_t i = 0; i < n; ++i) a.list.push_back(read_arg());
    } else {
        a.kind = tf::Arg::N;
    }
    return a;
}
}  // namespace

namespace tf {
void add_module(const char* name, void (*init)(Module&)) { init(modules()[name]); }
}  // namespace tf

namespace at::cuda {
CUDAStream getCurrentCUDAStream(int) { return {reinterpret_cast<cudaStream_t>(0x57)}; }
cudaDeviceProp* getCurrentDeviceProperties() { return &props; }
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
    print_launch(f, g, b, smem, dim3(0, 0, 0), 0, args);
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
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* f, void** args) {
    dim3 c(0, 0, 0);
    int pdl = 0;
    for (unsigned i = 0; i < config->numAttrs; ++i) {
        const cudaLaunchAttribute& a = config->attrs[i];
        if (a.id == cudaLaunchAttributeClusterDimension)
            c = dim3(a.val.clusterDim.x, a.val.clusterDim.y, a.val.clusterDim.z);
        else if (a.id == cudaLaunchAttributeProgrammaticStreamSerialization)
            pdl = a.val.programmaticStreamSerializationAllowed;
        else
            throw std::runtime_error("an unknown launch attribute");
    }
    print_launch(f, config->gridDim, config->blockDim, config->dynamicSmemBytes, c, pdl, args);
    return cudaSuccess;
}
cudaError_t cudaFuncSetAttribute(const void* f, cudaFuncAttribute attr, int value) {
    printf("ATTR %d %s %d %d\n", call_no, name_of(f).c_str(), (int)attr, value);
    return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, const void* f, int threads, size_t smem) {
    printf("OCCQ %d %s %d %zu\n", call_no, name_of(f).c_str(), threads, smem);
    *n = occupancy;
    return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* n, const void* f, int threads, size_t smem,
                                                                   unsigned) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, f, threads, smem);
}
cudaError_t cudaGetLastError(void) { return cudaSuccess; }
cudaError_t cudaPeekAtLastError(void) { return cudaSuccess; }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: glm_mm_launches params.txt < calls.txt\n");
        return 2;
    }
    std::ifstream pt(argv[1]);
    std::string k;
    int n;
    while (pt >> k >> n) {
        std::vector<int> sizes(n);
        for (int& s : sizes) pt >> s;
        table[k] = sizes;
    }
    std::string word;
    while (std::cin >> word) {
        if (word == "CASE") {
            std::string id;
            std::cin >> id;
            call_no = -1;
            printf("CASE %s\n", id.c_str());
        } else if (word == "CHIP") {
            int optin;
            std::cin >> props.major >> props.minor >> props.multiProcessorCount >> optin;
            props.sharedMemPerBlockOptin = (size_t)optin;
        } else if (word == "OCC") {
            std::cin >> occupancy;
        } else if (word == "CALL") {
            std::string mod, fn;
            size_t nargs;
            std::cin >> mod >> fn >> nargs;
            ++call_no;
            std::vector<tf::Arg> args(nargs);
            for (auto& a : args) a = read_arg();
            try {
                auto m = modules().find(mod);
                if (m == modules().end()) throw std::runtime_error("no module " + mod);
                auto f = m->second.fns.find(fn);
                if (f == m->second.fns.end()) throw std::runtime_error("no function " + fn);
                f->second(args);
            } catch (const std::exception& e) {
                printf("ERROR %d %s\n", call_no, e.what());
            }
        }
        fflush(stdout);
    }
    return 0;
}
