#ifndef LUNAFLUX_CUDA_ABI_H
#define LUNAFLUX_CUDA_ABI_H

#include <moonbit.h>

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE lf_library;
typedef INIT_ONCE lf_once;
#define LF_ONCE_INIT INIT_ONCE_STATIC_INIT
#else
#include <dlfcn.h>
#include <pthread.h>
typedef void *lf_library;
typedef pthread_once_t lf_once;
#define LF_ONCE_INIT PTHREAD_ONCE_INIT
#endif

typedef int32_t CUdevice;
typedef int32_t CUresult;
typedef struct CUuuid_st { char bytes[16]; } CUuuid;
#define CUDA_ERROR_NOT_READY 600
typedef uint64_t CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;

enum lf_availability {
  LF_AVAILABLE = 0,
  LF_UNSUPPORTED_PLATFORM = 1,
  LF_DRIVER_LIBRARY_MISSING = 2,
  LF_DRIVER_ABI_INCOMPLETE = 3,
  LF_DRIVER_INITIALIZATION_FAILED = 4,
  LF_NO_DEVICES = 5,
  LF_INVENTORY_FAILED = 6
};

enum lf_status {
  LF_OK = 0,
  LF_UNAVAILABLE = -1,
  LF_INVALID_ARGUMENT = -2,
  LF_CLOSED = -3,
  LF_BUSY = -4,
  LF_DRIVER_FAILURE = -5,
  LF_HOST_ALLOCATION_FAILED = -6,
  LF_SIZE_OVERFLOW = -7,
  LF_UNSUPPORTED = -8,
  LF_INVALID_OUTPUT = -9
};

typedef struct lf_cuda_api {
  lf_library driver_library;
  lf_library cublas_library;
  int32_t availability;
  int32_t driver_version;
  int32_t device_count;
  int32_t cublas_available;
  int32_t graph_available;
  CUresult (*cuInit)(uint32_t);
  CUresult (*cuDriverGetVersion)(int32_t *);
  CUresult (*cuDeviceGetCount)(int32_t *);
  CUresult (*cuDeviceGet)(CUdevice *, int32_t);
  CUresult (*cuDeviceGetUuid)(CUuuid *, CUdevice);
  CUresult (*cuDeviceCanAccessPeer)(int32_t *, CUdevice, CUdevice);
  CUresult (*cuDeviceGetName)(char *, int32_t, CUdevice);
  CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
  CUresult (*cuDeviceGetAttribute)(int32_t *, int32_t, CUdevice);
  CUresult (*cuCtxCreate)(CUcontext *, uint32_t, CUdevice);
  CUresult (*cuCtxDestroy)(CUcontext);
  CUresult (*cuCtxSetCurrent)(CUcontext);
  CUresult (*cuStreamCreate)(CUstream *, uint32_t);
  CUresult (*cuStreamDestroy)(CUstream);
  CUresult (*cuStreamQuery)(CUstream);
  CUresult (*cuStreamSynchronize)(CUstream);
  CUresult (*cuEventCreate)(CUevent *, uint32_t);
  CUresult (*cuEventDestroy)(CUevent);
  CUresult (*cuEventRecord)(CUevent, CUstream);
  CUresult (*cuEventQuery)(CUevent);
  CUresult (*cuEventSynchronize)(CUevent);
  CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent);
  CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
  CUresult (*cuMemFree)(CUdeviceptr);
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
  CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
  CUresult (*cuModuleLoadData)(CUmodule *, const void *);
  CUresult (*cuModuleUnload)(CUmodule);
  CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
  CUresult (*cuLaunchKernel)(
    CUfunction,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    CUstream,
    void **,
    void **
  );
  CUresult (*cuStreamBeginCapture)(CUstream, int32_t);
  CUresult (*cuStreamEndCapture)(CUstream, CUgraph *);
  CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec *, CUgraph, uint64_t);
  CUresult (*cuGraphDestroy)(CUgraph);
  CUresult (*cuGraphExecDestroy)(CUgraphExec);
  CUresult (*cuGraphLaunch)(CUgraphExec, CUstream);
  int32_t (*cublasLtCreate)(cublasLtHandle_t *);
  int32_t (*cublasLtDestroy)(cublasLtHandle_t);
  int32_t (*cublasLtMatmulDescCreate)(
    cublasLtMatmulDesc_t *,
    int32_t,
    int32_t
  );
  int32_t (*cublasLtMatmulDescSetAttribute)(
    cublasLtMatmulDesc_t,
    int32_t,
    const void *,
    size_t
  );
  int32_t (*cublasLtMatmulDescDestroy)(cublasLtMatmulDesc_t);
  int32_t (*cublasLtMatrixLayoutCreate)(
    cublasLtMatrixLayout_t *,
    int32_t,
    uint64_t,
    uint64_t,
    int64_t
  );
  int32_t (*cublasLtMatrixLayoutDestroy)(cublasLtMatrixLayout_t);
  int32_t (*cublasLtMatmul)(
    cublasLtHandle_t,
    cublasLtMatmulDesc_t,
    const void *,
    const void *,
    cublasLtMatrixLayout_t,
    const void *,
    cublasLtMatrixLayout_t,
    const void *,
    const void *,
    cublasLtMatrixLayout_t,
    void *,
    cublasLtMatrixLayout_t,
    const void *,
    void *,
    size_t,
    CUstream
  );
} lf_cuda_api;

lf_cuda_api *lf_cuda_api_get(void);
int32_t lf_cuda_map_result(CUresult result);
int32_t lf_cuda_device_can_access_peer_with_api(
  lf_cuda_api *api,
  int32_t source_ordinal,
  int32_t destination_ordinal,
  int32_t *output
);

#endif
