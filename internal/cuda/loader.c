#include "cuda_abi.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID 33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID 34
#define CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID 50

static lf_cuda_api loader_api;
static lf_once loader_once = LF_ONCE_INIT;

/* Driver and cuBLASLt libraries intentionally remain loaded for process
 * lifetime. The immutable dispatch table is published once, and unloading it
 * would invalidate function pointers while contexts or child resources may
 * still be alive. The bounded library handles are therefore loader state, not
 * per-engine resources. */

#if !defined(__APPLE__)
#if defined(_WIN32)
static lf_library lf_open_library(const char *name) {
  return LoadLibraryA(name);
}

static void *lf_symbol(lf_library library, const char *name) {
  return (void *)GetProcAddress(library, name);
}
#else
static lf_library lf_open_library(const char *name) {
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

static void *lf_symbol(lf_library library, const char *name) {
  return dlsym(library, name);
}
#endif

static void lf_load_optional_cublas(lf_cuda_api *api) {
  static const char *names[] = {
#if defined(_WIN32)
    "cublasLt64_13.dll",
    "cublasLt64_12.dll",
    "cublasLt64_11.dll"
#else
    "libcublasLt.so.13",
    "libcublasLt.so.12",
    "libcublasLt.so.11",
    "libcublasLt.so"
#endif
  };
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i += 1) {
    api->cublas_library = lf_open_library(names[i]);
    if (api->cublas_library != NULL) break;
  }
  if (api->cublas_library == NULL) return;
  api->cublasLtCreate = (int32_t (*)(cublasLtHandle_t *))
    lf_symbol(api->cublas_library, "cublasLtCreate");
  api->cublasLtDestroy = (int32_t (*)(cublasLtHandle_t))
    lf_symbol(api->cublas_library, "cublasLtDestroy");
  api->cublasLtMatmulDescCreate =
    (int32_t (*)(cublasLtMatmulDesc_t *, int32_t, int32_t))
      lf_symbol(api->cublas_library, "cublasLtMatmulDescCreate");
  api->cublasLtMatmulDescSetAttribute =
    (int32_t (*)(cublasLtMatmulDesc_t, int32_t, const void *, size_t))
      lf_symbol(api->cublas_library, "cublasLtMatmulDescSetAttribute");
  api->cublasLtMatmulDescDestroy =
    (int32_t (*)(cublasLtMatmulDesc_t))
      lf_symbol(api->cublas_library, "cublasLtMatmulDescDestroy");
  api->cublasLtMatrixLayoutCreate =
    (int32_t (*)(cublasLtMatrixLayout_t *, int32_t, uint64_t, uint64_t,
                 int64_t))
      lf_symbol(api->cublas_library, "cublasLtMatrixLayoutCreate");
  api->cublasLtMatrixLayoutDestroy =
    (int32_t (*)(cublasLtMatrixLayout_t))
      lf_symbol(api->cublas_library, "cublasLtMatrixLayoutDestroy");
  api->cublasLtMatmul =
    (int32_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *,
                 const void *, cublasLtMatrixLayout_t, const void *,
                 cublasLtMatrixLayout_t, const void *, const void *,
                 cublasLtMatrixLayout_t, void *, cublasLtMatrixLayout_t,
                 const void *, void *, size_t, CUstream))
      lf_symbol(api->cublas_library, "cublasLtMatmul");
  api->cublas_available =
    api->cublasLtCreate != NULL && api->cublasLtDestroy != NULL &&
    api->cublasLtMatmulDescCreate != NULL &&
    api->cublasLtMatmulDescSetAttribute != NULL &&
    api->cublasLtMatmulDescDestroy != NULL &&
    api->cublasLtMatrixLayoutCreate != NULL &&
    api->cublasLtMatrixLayoutDestroy != NULL &&
    api->cublasLtMatmul != NULL;
}

static void lf_load_optional_graph(lf_cuda_api *api) {
  api->cuStreamBeginCapture =
    (CUresult (*)(CUstream, int32_t))
      lf_symbol(api->driver_library, "cuStreamBeginCapture");
  api->cuStreamEndCapture =
    (CUresult (*)(CUstream, CUgraph *))
      lf_symbol(api->driver_library, "cuStreamEndCapture");
  api->cuGraphInstantiateWithFlags =
    (CUresult (*)(CUgraphExec *, CUgraph, uint64_t))
      lf_symbol(api->driver_library, "cuGraphInstantiateWithFlags");
  api->cuGraphDestroy =
    (CUresult (*)(CUgraph))lf_symbol(api->driver_library, "cuGraphDestroy");
  api->cuGraphExecDestroy =
    (CUresult (*)(CUgraphExec))
      lf_symbol(api->driver_library, "cuGraphExecDestroy");
  api->cuGraphLaunch =
    (CUresult (*)(CUgraphExec, CUstream))
      lf_symbol(api->driver_library, "cuGraphLaunch");
  api->graph_available =
    api->cuStreamBeginCapture != NULL && api->cuStreamEndCapture != NULL &&
    api->cuGraphInstantiateWithFlags != NULL && api->cuGraphDestroy != NULL &&
    api->cuGraphExecDestroy != NULL && api->cuGraphLaunch != NULL;
}
#endif

#define LF_LOAD_REQUIRED(field, symbol_name)                                   \
  do {                                                                         \
    api->field = (void *)lf_symbol(api->driver_library, symbol_name);          \
    if (api->field == NULL) {                                                   \
      api->availability = LF_DRIVER_ABI_INCOMPLETE;                            \
      return;                                                                  \
    }                                                                          \
  } while (0)

static void lf_initialize_loader(void) {
  lf_cuda_api *api = &loader_api;
  memset(api, 0, sizeof(*api));
#if defined(__APPLE__)
  api->availability = LF_UNSUPPORTED_PLATFORM;
  return;
#else
  static const char *driver_names[] = {
#if defined(_WIN32)
    "nvcuda.dll"
#else
    "libcuda.so.1",
    "libcuda.so"
#endif
  };
  for (size_t i = 0;
       i < sizeof(driver_names) / sizeof(driver_names[0]);
       i += 1) {
    api->driver_library = lf_open_library(driver_names[i]);
    if (api->driver_library != NULL) break;
  }
  if (api->driver_library == NULL) {
    api->availability = LF_DRIVER_LIBRARY_MISSING;
    return;
  }
  LF_LOAD_REQUIRED(cuInit, "cuInit");
  LF_LOAD_REQUIRED(cuDriverGetVersion, "cuDriverGetVersion");
  LF_LOAD_REQUIRED(cuDeviceGetCount, "cuDeviceGetCount");
  LF_LOAD_REQUIRED(cuDeviceGet, "cuDeviceGet");
  api->cuDeviceGetUuid = (CUresult (*)(CUuuid *, CUdevice))
    lf_symbol(api->driver_library, "cuDeviceGetUuid_v2");
  if (api->cuDeviceGetUuid == NULL) {
    api->cuDeviceGetUuid = (CUresult (*)(CUuuid *, CUdevice))
      lf_symbol(api->driver_library, "cuDeviceGetUuid");
  }
  if (api->cuDeviceGetUuid == NULL) {
    api->availability = LF_DRIVER_ABI_INCOMPLETE;
    return;
  }
  LF_LOAD_REQUIRED(cuDeviceCanAccessPeer, "cuDeviceCanAccessPeer");
  LF_LOAD_REQUIRED(cuDeviceGetName, "cuDeviceGetName");
  LF_LOAD_REQUIRED(cuDeviceTotalMem, "cuDeviceTotalMem_v2");
  LF_LOAD_REQUIRED(cuDeviceGetAttribute, "cuDeviceGetAttribute");
  LF_LOAD_REQUIRED(cuCtxCreate, "cuCtxCreate_v2");
  LF_LOAD_REQUIRED(cuCtxDestroy, "cuCtxDestroy_v2");
  LF_LOAD_REQUIRED(cuCtxSetCurrent, "cuCtxSetCurrent");
  LF_LOAD_REQUIRED(cuStreamCreate, "cuStreamCreate");
  LF_LOAD_REQUIRED(cuStreamDestroy, "cuStreamDestroy_v2");
  LF_LOAD_REQUIRED(cuStreamQuery, "cuStreamQuery");
  LF_LOAD_REQUIRED(cuStreamSynchronize, "cuStreamSynchronize");
  LF_LOAD_REQUIRED(cuEventCreate, "cuEventCreate");
  LF_LOAD_REQUIRED(cuEventDestroy, "cuEventDestroy_v2");
  LF_LOAD_REQUIRED(cuEventRecord, "cuEventRecord");
  LF_LOAD_REQUIRED(cuEventQuery, "cuEventQuery");
  LF_LOAD_REQUIRED(cuEventSynchronize, "cuEventSynchronize");
  LF_LOAD_REQUIRED(cuEventElapsedTime, "cuEventElapsedTime");
  LF_LOAD_REQUIRED(cuMemAlloc, "cuMemAlloc_v2");
  LF_LOAD_REQUIRED(cuMemFree, "cuMemFree_v2");
  LF_LOAD_REQUIRED(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
  LF_LOAD_REQUIRED(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
  LF_LOAD_REQUIRED(cuModuleLoadData, "cuModuleLoadData");
  LF_LOAD_REQUIRED(cuModuleUnload, "cuModuleUnload");
  LF_LOAD_REQUIRED(cuModuleGetFunction, "cuModuleGetFunction");
  LF_LOAD_REQUIRED(cuLaunchKernel, "cuLaunchKernel");
  lf_load_optional_graph(api);
  if (api->cuInit(0) != CUDA_SUCCESS) {
    api->availability = LF_DRIVER_INITIALIZATION_FAILED;
    return;
  }
  if (api->cuDriverGetVersion(&api->driver_version) != CUDA_SUCCESS ||
      api->cuDeviceGetCount(&api->device_count) != CUDA_SUCCESS) {
    api->availability = LF_INVENTORY_FAILED;
    return;
  }
  if (api->device_count <= 0) {
    api->availability = LF_NO_DEVICES;
    return;
  }
  lf_load_optional_cublas(api);
  api->availability = LF_AVAILABLE;
#endif
}

#if defined(_WIN32)
static BOOL CALLBACK lf_initialize_loader_once(
  PINIT_ONCE once,
  PVOID parameter,
  PVOID *context
) {
  (void)once;
  (void)parameter;
  (void)context;
  lf_initialize_loader();
  return TRUE;
}
#endif

lf_cuda_api *lf_cuda_api_get(void) {
#if defined(_WIN32)
  InitOnceExecuteOnce(
    &loader_once,
    lf_initialize_loader_once,
    NULL,
    NULL
  );
#else
  pthread_once(&loader_once, lf_initialize_loader);
#endif
  return &loader_api;
}

int32_t lf_cuda_map_result(CUresult result) {
  return result == CUDA_SUCCESS ? LF_OK : LF_DRIVER_FAILURE;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_probe(int64_t *output) {
  lf_cuda_api *api = lf_cuda_api_get();
  output[0] = api->driver_version;
  output[1] = api->device_count;
  output[2] = api->cublas_available;
  return api->availability;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_device_info(
  int32_t ordinal,
  int64_t *numeric,
  uint8_t *name,
  uint8_t *uuid,
  uint8_t *pci
) {
  lf_cuda_api *api = lf_cuda_api_get();
  if (api->availability != LF_AVAILABLE) return LF_UNAVAILABLE;
  if (ordinal < 0 || ordinal >= api->device_count) return LF_INVALID_ARGUMENT;
  CUdevice device = 0;
  size_t total_memory = 0;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t pci_domain = 0;
  int32_t pci_bus = 0;
  int32_t pci_device = 0;
  CUuuid local_uuid;
  char local_name[96];
  memset(local_name, 0, sizeof(local_name));
  memset(&local_uuid, 0, sizeof(local_uuid));
  if (api->cuDeviceGet(&device, ordinal) != CUDA_SUCCESS ||
      api->cuDeviceTotalMem(&total_memory, device) != CUDA_SUCCESS ||
      api->cuDeviceGetAttribute(
        &major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device
      ) != CUDA_SUCCESS ||
      api->cuDeviceGetAttribute(
        &minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device
      ) != CUDA_SUCCESS ||
      api->cuDeviceGetAttribute(
        &pci_domain,
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
        device
      ) != CUDA_SUCCESS ||
      api->cuDeviceGetAttribute(
        &pci_bus,
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        device
      ) != CUDA_SUCCESS ||
      api->cuDeviceGetAttribute(
        &pci_device,
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        device
      ) != CUDA_SUCCESS ||
      api->cuDeviceGetUuid(&local_uuid, device) != CUDA_SUCCESS ||
      api->cuDeviceGetName(local_name, (int32_t)sizeof(local_name), device) !=
        CUDA_SUCCESS) {
    return LF_DRIVER_FAILURE;
  }
  numeric[0] = (int64_t)total_memory;
  numeric[1] = major;
  numeric[2] = minor;
  numeric[3] = major >= 8;
  numeric[4] = api->cublas_available;
  memcpy(name, local_name, sizeof(local_name));
  name[95] = 0;
  (void)snprintf(
    (char *)uuid,
    41,
    "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
    (unsigned char)local_uuid.bytes[0], (unsigned char)local_uuid.bytes[1],
    (unsigned char)local_uuid.bytes[2], (unsigned char)local_uuid.bytes[3],
    (unsigned char)local_uuid.bytes[4], (unsigned char)local_uuid.bytes[5],
    (unsigned char)local_uuid.bytes[6], (unsigned char)local_uuid.bytes[7],
    (unsigned char)local_uuid.bytes[8], (unsigned char)local_uuid.bytes[9],
    (unsigned char)local_uuid.bytes[10], (unsigned char)local_uuid.bytes[11],
    (unsigned char)local_uuid.bytes[12], (unsigned char)local_uuid.bytes[13],
    (unsigned char)local_uuid.bytes[14], (unsigned char)local_uuid.bytes[15]
  );
  (void)snprintf(
    (char *)pci,
    17,
    "%08x:%02x:%02x.0",
    (unsigned int)pci_domain,
    (unsigned int)pci_bus,
    (unsigned int)pci_device
  );
  return LF_OK;
}
