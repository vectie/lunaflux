#include "nccl_abi.h"

#include <string.h>

#define NCCL_SUCCESS 0

/* The admitted DSO and immutable dispatch table are bounded process-lifetime
 * loader state. They are not a per-engine context. Unloading would invalidate
 * function pointers while a communicator may still own native authority. */
#if defined(__linux__)
static lf_nccl_api loader_api;
static lf_nccl_once loader_once = LF_NCCL_ONCE_INIT;

static void lf_nccl_initialize_loader(void) {
  lf_nccl_api *api = &loader_api;
  memset(api, 0, sizeof(*api));
  api->library = dlopen("libnccl.so.2", RTLD_NOW | RTLD_LOCAL);
  if (api->library == NULL) {
    api->availability = LF_NCCL_LIBRARY_MISSING;
    return;
  }
#define LF_NCCL_LOAD_REQUIRED(field, name)                                    \
  do {                                                                        \
    api->field = (void *)dlsym(api->library, name);                           \
    if (api->field == NULL) {                                                  \
      api->availability = LF_NCCL_ABI_INCOMPLETE;                            \
      return;                                                                 \
    }                                                                         \
  } while (0)
  LF_NCCL_LOAD_REQUIRED(get_version, "ncclGetVersion");
  LF_NCCL_LOAD_REQUIRED(get_unique_id, "ncclGetUniqueId");
  LF_NCCL_LOAD_REQUIRED(comm_init_rank_config, "ncclCommInitRankConfig");
  LF_NCCL_LOAD_REQUIRED(comm_get_async_error, "ncclCommGetAsyncError");
  LF_NCCL_LOAD_REQUIRED(comm_destroy, "ncclCommDestroy");
  LF_NCCL_LOAD_REQUIRED(comm_abort, "ncclCommAbort");
  LF_NCCL_LOAD_REQUIRED(all_reduce, "ncclAllReduce");
  LF_NCCL_LOAD_REQUIRED(all_gather, "ncclAllGather");
#undef LF_NCCL_LOAD_REQUIRED
  if (api->get_version(&api->version_code) != NCCL_SUCCESS ||
      api->version_code <= 0) {
    api->version_code = 0;
    api->availability = LF_NCCL_VERSION_UNAVAILABLE;
    return;
  }
  api->availability = LF_NCCL_AVAILABLE;
}

lf_nccl_api *lf_nccl_api_get(void) {
  pthread_once(&loader_once, lf_nccl_initialize_loader);
  return &loader_api;
}
#else
static lf_nccl_api loader_api = {
  .availability = LF_NCCL_UNSUPPORTED_PLATFORM
};

lf_nccl_api *lf_nccl_api_get(void) {
  return &loader_api;
}
#endif

int32_t lf_nccl_runtime_admit_with_api(
  lf_nccl_api *api,
  int32_t minimum_version,
  int32_t maximum_version,
  int32_t *version
) {
  if (api == NULL || version == NULL || minimum_version <= 0 ||
      maximum_version < minimum_version) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  *version = api->version_code;
  if (api->availability != LF_NCCL_AVAILABLE) return LF_NCCL_UNAVAILABLE;
  if (api->version_code < minimum_version ||
      api->version_code > maximum_version) {
    return LF_NCCL_VERSION_MISMATCH;
  }
  return LF_NCCL_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_probe(int32_t *version) {
  lf_nccl_api *api = lf_nccl_api_get();
  if (version != NULL) *version = api->version_code;
  return api->availability;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_runtime_admit(
  int32_t minimum_version,
  int32_t maximum_version,
  int32_t *version
) {
  return lf_nccl_runtime_admit_with_api(
    lf_nccl_api_get(),
    minimum_version,
    maximum_version,
    version
  );
}

int32_t lf_nccl_unique_id_create_with_api(
  lf_nccl_api *api,
  int32_t admitted_version,
  uint8_t *output
) {
  if (output == NULL || admitted_version <= 0) return LF_NCCL_INVALID_ARGUMENT;
  if (api->availability != LF_NCCL_AVAILABLE) return LF_NCCL_UNAVAILABLE;
  if (api->version_code != admitted_version) return LF_NCCL_VERSION_MISMATCH;
  lf_nccl_unique_id id;
  memset(&id, 0, sizeof(id));
  if (api->get_unique_id(&id) != NCCL_SUCCESS) {
    return LF_NCCL_RUNTIME_FAILURE;
  }
  memcpy(output, &id, sizeof(id));
  return LF_NCCL_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_unique_id_create(
  int32_t admitted_version,
  uint8_t *output
) {
  return lf_nccl_unique_id_create_with_api(
    lf_nccl_api_get(),
    admitted_version,
    output
  );
}
