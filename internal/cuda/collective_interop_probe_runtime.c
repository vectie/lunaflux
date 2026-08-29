#include "resource_internal.h"

lf_cuda_api *lf_cuda_api_get(void) {
  return NULL;
}

int32_t lf_cuda_map_result(CUresult result) {
  return result == 0 ? LF_OK : LF_DRIVER_FAILURE;
}
