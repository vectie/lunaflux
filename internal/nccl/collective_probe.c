#include "nccl_abi.h"

#include "../cuda/collective_interop_probe.h"

#include <stdint.h>
#include <string.h>

#define LF_FAKE_VERSION 22005

typedef struct {
  int32_t all_reduce_calls;
  int32_t all_gather_calls;
  int32_t abort_calls;
  int32_t datatype;
  int32_t reduction;
  size_t count;
  const void *send;
  void *receive;
} lf_collective_probe_state;

static lf_collective_probe_state *active_collective_probe = NULL;

static lf_nccl_result lf_probe_init(
  lf_nccl_handle *handle,
  int32_t world_size,
  lf_nccl_unique_id id,
  int32_t rank,
  lf_nccl_config_v21400 *config
) {
  (void)id;
  if (world_size != 2 || rank != 1 || config == NULL ||
      config->blocking != 0) return 1;
  *handle = (void *)(uintptr_t)0x4000;
  return 0;
}

static lf_nccl_result lf_probe_async(
  lf_nccl_handle handle,
  lf_nccl_result *status
) {
  if (handle != (void *)(uintptr_t)0x4000 || status == NULL) return 1;
  *status = 0;
  return 0;
}

static lf_nccl_result lf_probe_destroy(lf_nccl_handle handle) {
  return handle == (void *)(uintptr_t)0x4000 ? 0 : 1;
}

static lf_nccl_result lf_probe_abort(lf_nccl_handle handle) {
  if (active_collective_probe == NULL ||
      handle != (void *)(uintptr_t)0x4000) {
    return 1;
  }
  active_collective_probe->abort_calls += 1;
  return 0;
}

static lf_nccl_result lf_probe_all_reduce(
  const void *send,
  void *receive,
  size_t count,
  int32_t datatype,
  int32_t reduction,
  lf_nccl_handle communicator,
  lf_nccl_stream stream
) {
  if (active_collective_probe == NULL ||
      communicator != (void *)(uintptr_t)0x4000 ||
      stream != (void *)(uintptr_t)0x3000) {
    return 1;
  }
  active_collective_probe->all_reduce_calls += 1;
  active_collective_probe->send = send;
  active_collective_probe->receive = receive;
  active_collective_probe->count = count;
  active_collective_probe->datatype = datatype;
  active_collective_probe->reduction = reduction;
  return 0;
}

static lf_nccl_result lf_probe_all_gather(
  const void *send,
  void *receive,
  size_t count,
  int32_t datatype,
  lf_nccl_handle communicator,
  lf_nccl_stream stream
) {
  if (active_collective_probe == NULL ||
      communicator != (void *)(uintptr_t)0x4000 ||
      stream != (void *)(uintptr_t)0x3000) {
    return 1;
  }
  active_collective_probe->all_gather_calls += 1;
  active_collective_probe->send = send;
  active_collective_probe->receive = receive;
  active_collective_probe->count = count;
  active_collective_probe->datatype = datatype;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_test_collectives(int32_t cycles) {
  if (cycles <= 0 || cycles > 100000) return LF_NCCL_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    lf_collective_probe_state state;
    memset(&state, 0, sizeof(state));
    active_collective_probe = &state;
    lf_nccl_api api;
    memset(&api, 0, sizeof(api));
    api.availability = LF_NCCL_AVAILABLE;
    api.version_code = LF_FAKE_VERSION;
    api.comm_init_rank_config = lf_probe_init;
    api.comm_get_async_error = lf_probe_async;
    api.comm_destroy = lf_probe_destroy;
    api.comm_abort = lf_probe_abort;
    api.all_reduce = lf_probe_all_reduce;
    api.all_gather = lf_probe_all_gather;
    uint8_t id[LF_NCCL_UNIQUE_ID_BYTES];
    memset(id, 0x5a, sizeof(id));
    lf_device_interop_managed_probe device;
    lf_device_interop_managed_probe_init(&device);
    int32_t status = 0;
    lf_nccl_communicator *communicator =
      lf_nccl_communicator_create_on_context_with_api(
        &api,
        LF_FAKE_VERSION,
        id,
        2,
        1,
        (uint64_t)cycle + 1U,
        0U,
        lf_device_interop_managed_probe_context(&device),
        &status
      );
    int32_t ready = 0;
    if (status != LF_NCCL_OK ||
        lf_nccl_communicator_poll_ready(communicator, &ready) != LF_NCCL_OK ||
        ready != 1 ||
        lf_nccl_communicator_submit_bf16(
          communicator,
          (uint64_t)cycle + 1U,
          1U,
          1U,
          0,
          1,
          lf_device_interop_managed_probe_context(&device),
          lf_device_interop_managed_probe_send(&device),
          0,
          48,
          lf_device_interop_managed_probe_receive(&device),
          0,
          48,
          lf_device_interop_managed_probe_queue(&device)
        ) != LF_NCCL_OK ||
        state.all_reduce_calls != 1 ||
        state.count != 48U || state.datatype != 9 || state.reduction != 0 ||
        state.send != (void *)(uintptr_t)0x10000 ||
        state.receive != (void *)(uintptr_t)0x20000) {
      return 10;
    }
    int32_t completed = 0;
    if (lf_nccl_communicator_poll_collective(
          communicator,
          &completed
        ) != LF_NCCL_OK || completed != 1 ||
        lf_device_interop_managed_probe_query_calls(&device) != 1 ||
        communicator->next_collective_sequence != 2U) return 11;
    if (lf_nccl_communicator_submit_bf16(
          communicator,
          (uint64_t)cycle + 1U,
          1U,
          2U,
          10,
          2,
          lf_device_interop_managed_probe_context(&device),
          lf_device_interop_managed_probe_send(&device),
          32,
          24,
          lf_device_interop_managed_probe_receive(&device),
          64,
          48,
          lf_device_interop_managed_probe_queue(&device)
        ) != LF_NCCL_OK ||
        state.all_gather_calls != 1 ||
        lf_device_interop_managed_probe_query_calls(&device) != 1 ||
        state.count != 24U || state.datatype != 9 ||
        state.send != (void *)(uintptr_t)0x10020 ||
        state.receive != (void *)(uintptr_t)0x20040 ||
        communicator->next_collective_sequence != 2U) {
      return 12;
    }
    completed = 0;
    if (lf_nccl_communicator_poll_collective(
          communicator,
          &completed
        ) != LF_NCCL_OK || completed != 1 ||
        lf_device_interop_managed_probe_query_calls(&device) != 2 ||
        communicator->next_collective_sequence != 3U) return 13;
    if (lf_nccl_communicator_close(communicator) != LF_NCCL_OK) return 14;
    moonbit_decref(communicator);
    lf_device_interop_managed_probe_close(&device);
  }
  active_collective_probe = NULL;
  return LF_NCCL_OK;
}
