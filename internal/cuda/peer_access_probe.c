#include "cuda_abi.h"

#include <stdint.h>
#include <string.h>

typedef struct lf_peer_probe_state {
  int32_t get_calls;
  int32_t peer_calls;
  int32_t fail_get_ordinal;
  int32_t fail_peer;
  int32_t malformed_output;
  CUdevice last_source;
  CUdevice last_destination;
} lf_peer_probe_state;

static lf_peer_probe_state *lf_active_peer_probe;

static CUresult lf_peer_probe_device_get(CUdevice *device, int32_t ordinal) {
  lf_peer_probe_state *state = lf_active_peer_probe;
  state->get_calls += 1;
  if (ordinal == state->fail_get_ordinal) return 1;
  *device = ordinal + 100;
  return 0;
}

static CUresult lf_peer_probe_query(
  int32_t *output,
  CUdevice source,
  CUdevice destination
) {
  lf_peer_probe_state *state = lf_active_peer_probe;
  state->peer_calls += 1;
  state->last_source = source;
  state->last_destination = destination;
  if (state->fail_peer != 0) return 1;
  if (state->malformed_output != 0) {
    *output = 2;
  } else {
    *output = source < destination ? 1 : 0;
  }
  return 0;
}

static int32_t lf_expect_query(
  lf_cuda_api *api,
  lf_peer_probe_state *state,
  int32_t source,
  int32_t destination,
  int32_t expected_status,
  int32_t expected_output
) {
  int32_t output = 73;
  int32_t status = lf_cuda_device_can_access_peer_with_api(
    api,
    source,
    destination,
    &output
  );
  if (status != expected_status || output != expected_output) return 1;
  if (status == LF_OK &&
      (state->last_source != source + 100 ||
       state->last_destination != destination + 100)) {
    return 2;
  }
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_peer_access_boundary(int32_t cycles) {
  if (cycles <= 0 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    lf_cuda_api api;
    lf_peer_probe_state state;
    memset(&api, 0, sizeof(api));
    memset(&state, 0, sizeof(state));
    state.fail_get_ordinal = -1;
    api.availability = LF_AVAILABLE;
    api.device_count = 3;
    api.cuDeviceGet = lf_peer_probe_device_get;
    api.cuDeviceCanAccessPeer = lf_peer_probe_query;
    lf_active_peer_probe = &state;

    if (lf_cuda_device_can_access_peer_with_api(
          NULL,
          0,
          1,
          &state.last_source
        ) != LF_INVALID_ARGUMENT ||
        lf_cuda_device_can_access_peer_with_api(
          &api,
          0,
          1,
          NULL
        ) != LF_INVALID_ARGUMENT) {
      return 9;
    }

    if (lf_expect_query(&api, &state, 0, 1, LF_OK, 1) != 0 ||
        lf_expect_query(&api, &state, 1, 0, LF_OK, 0) != 0) {
      return 10;
    }
    if (state.get_calls != 4 || state.peer_calls != 2) return 10;
    int32_t calls = state.get_calls;
    int32_t peer_calls = state.peer_calls;
    if (lf_expect_query(&api, &state, -1, 0, LF_INVALID_ARGUMENT, 0) != 0 ||
        lf_expect_query(&api, &state, 0, 0, LF_INVALID_ARGUMENT, 0) != 0 ||
        lf_expect_query(&api, &state, 0, 3, LF_INVALID_ARGUMENT, 0) != 0 ||
        state.get_calls != calls || state.peer_calls != peer_calls) {
      return 11;
    }
    api.availability = LF_UNSUPPORTED_PLATFORM;
    if (lf_expect_query(&api, &state, 0, 1, LF_UNAVAILABLE, 0) != 0) {
      return 12;
    }
    api.availability = LF_AVAILABLE;
    state.fail_get_ordinal = 1;
    if (lf_expect_query(&api, &state, 0, 1, LF_DRIVER_FAILURE, 0) != 0) {
      return 13;
    }
    state.fail_get_ordinal = -1;
    state.fail_peer = 1;
    if (lf_expect_query(&api, &state, 0, 1, LF_DRIVER_FAILURE, 0) != 0) {
      return 14;
    }
    state.fail_peer = 0;
    state.malformed_output = 1;
    if (lf_expect_query(&api, &state, 0, 1, LF_INVALID_OUTPUT, 0) != 0) {
      return 15;
    }
    state.malformed_output = 0;
    api.cuDeviceCanAccessPeer = NULL;
    if (lf_expect_query(&api, &state, 0, 1, LF_UNAVAILABLE, 0) != 0) {
      return 16;
    }
  }
  return LF_OK;
}

#if defined(LUNAFLUX_CUDA_PEER_ACCESS_SANITIZER_MAIN)
lf_cuda_api *lf_cuda_api_get(void) {
  return NULL;
}

int main(void) {
  return lunaflux_cuda_test_peer_access_boundary(4096) == LF_OK ? 0 : 1;
}
#endif
