// cudaLaunchKernelEx for exl3_launches.cpp's stand-in runtime (qualification_6052b66d): python-0.6.6 with Mia's 0128 (MiaAI-Lab,
// GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold a687724, Apache-2.0) launches its expert kernels through launch_ex, i.e.
// cudaLaunchKernelEx -> cudaLaunchKernelExC. Each attribute comes out first as "LAUNCHATTR id value" (the value of
// cudaLaunchAttributeProgrammaticStreamSerialization, else -1), then the launch as cudaLaunchKernel's own LAUNCH line.
#include <cuda_runtime.h>

#include <cstdio>

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* c, const void* f, void** args) {
    for (unsigned i = 0; i < c->numAttrs; ++i) {
        const cudaLaunchAttribute& a = c->attrs[i];
        const int v = a.id == cudaLaunchAttributeProgrammaticStreamSerialization
                          ? (int)a.val.programmaticStreamSerializationAllowed
                          : -1;
        printf("LAUNCHATTR %d %d\n", (int)a.id, v);
    }
    return cudaLaunchKernel(f, c->gridDim, c->blockDim, args, c->dynamicSmemBytes, c->stream);
}
