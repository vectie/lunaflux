#include "cuda_abi.h"

#include <stdint.h>

#define CUDA_SUCCESS 0

int32_t lf_cuda_device_can_access_peer_with_api(
  lf_cuda_api *api,
  int32_t source_ordinal,
  int32_t destination_ordinal,
  int32_t *output
) {
  if (api == NULL || output == NULL) return LF_INVALID_ARGUMENT;
  *output = 0;
  if (api->availability != LF_AVAILABLE) return LF_UNAVAILABLE;
  if (source_ordinal < 0 || destination_ordinal < 0 ||
      source_ordinal == destination_ordinal ||
      source_ordinal >= api->device_count ||
      destination_ordinal >= api->device_count) {
    return LF_INVALID_ARGUMENT;
  }
  if (api->cuDeviceGet == NULL || api->cuDeviceCanAccessPeer == NULL) {
    return LF_UNAVAILABLE;
  }
  CUdevice source = 0;
  CUdevice destination = 0;
  if (api->cuDeviceGet(&source, source_ordinal) != CUDA_SUCCESS ||
      api->cuDeviceGet(&destination, destination_ordinal) != CUDA_SUCCESS) {
    return LF_DRIVER_FAILURE;
  }
  int32_t can_access = 0;
  if (api->cuDeviceCanAccessPeer(&can_access, source, destination) !=
      CUDA_SUCCESS) {
    return LF_DRIVER_FAILURE;
  }
  if (can_access != 0 && can_access != 1) return LF_INVALID_OUTPUT;
  *output = can_access;
  return LF_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_device_can_access_peer(
  int32_t source_ordinal,
  int32_t destination_ordinal,
  int32_t *output
) {
  return lf_cuda_device_can_access_peer_with_api(
    lf_cuda_api_get(),
    source_ordinal,
    destination_ordinal,
    output
  );
}
