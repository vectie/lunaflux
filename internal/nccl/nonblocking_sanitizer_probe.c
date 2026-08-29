#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../cuda/collective_interop_probe.h"

#include "communicator.c"
#include "collectives.c"

static void (*probe_finalizer)(void *) = NULL;
static int32_t probe_increfs = 0;
static int32_t probe_decrefs = 0;
static lf_nccl_result probe_async_result = 0;
static lf_nccl_result probe_async_status = 0;
static lf_nccl_result probe_submit_result = LF_NCCL_IN_PROGRESS;

void *moonbit_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
) {
  probe_finalizer = finalize;
  return calloc(1, payload_size);
}

void moonbit_incref(void *object) {
  assert(object != NULL);
  probe_increfs += 1;
}

void moonbit_decref(void *object) {
  assert(object != NULL);
  probe_decrefs += 1;
}

lf_nccl_api *lf_nccl_api_get(void) {
  return NULL;
}

static lf_nccl_result probe_init(
  lf_nccl_handle *handle,
  int32_t world_size,
  lf_nccl_unique_id id,
  int32_t rank,
  lf_nccl_config_v21400 *config
) {
  (void)id;
  assert(world_size == 2 && rank == 1);
  assert(config != NULL && config->blocking == 0);
  *handle = (void *)(uintptr_t)0x4000;
  return LF_NCCL_IN_PROGRESS;
}

static lf_nccl_result probe_async(
  lf_nccl_handle handle,
  lf_nccl_result *status
) {
  assert(handle == (void *)(uintptr_t)0x4000 && status != NULL);
  *status = probe_async_status;
  return probe_async_result;
}

static lf_nccl_result probe_consume(lf_nccl_handle handle) {
  assert(handle == (void *)(uintptr_t)0x4000);
  return 0;
}

static lf_nccl_result probe_all_reduce(
  const void *send,
  void *receive,
  size_t count,
  int32_t datatype,
  int32_t reduction,
  lf_nccl_handle communicator,
  lf_nccl_stream queue
) {
  assert(send != NULL && receive != NULL && count == 16U);
  assert(datatype == 9 && reduction == 0);
  assert(communicator == (void *)(uintptr_t)0x4000);
  assert(queue == (void *)(uintptr_t)0x3000);
  return probe_submit_result;
}

static lf_nccl_result probe_all_gather(
  const void *send,
  void *receive,
  size_t count,
  int32_t datatype,
  lf_nccl_handle communicator,
  lf_nccl_stream queue
) {
  (void)send;
  (void)receive;
  (void)count;
  (void)datatype;
  (void)communicator;
  (void)queue;
  return LF_NCCL_IN_PROGRESS;
}

static void initialize_api(lf_nccl_api *api) {
  memset(api, 0, sizeof(*api));
  api->availability = LF_NCCL_AVAILABLE;
  api->version_code = 21403;
  api->comm_init_rank_config = probe_init;
  api->comm_get_async_error = probe_async;
  api->comm_destroy = probe_consume;
  api->comm_abort = probe_consume;
  api->all_reduce = probe_all_reduce;
  api->all_gather = probe_all_gather;
}

static lf_nccl_communicator *create_ready(
  lf_nccl_api *api,
  lf_device_interop_probe *device,
  uint64_t generation
) {
  uint8_t id[LF_NCCL_UNIQUE_ID_BYTES];
  memset(id, 0x5a, sizeof(id));
  int32_t status = 0;
  lf_nccl_communicator *communicator =
    lf_nccl_communicator_create_on_context_with_api(
      api,
      21403,
      id,
      2,
      1,
      generation,
      0U,
      lf_device_interop_probe_context(device),
      &status
    );
  assert(status == LF_NCCL_OK);
  int32_t ready = 1;
  probe_async_status = LF_NCCL_IN_PROGRESS;
  assert(lf_nccl_communicator_poll_ready(communicator, &ready) == LF_NCCL_OK);
  assert(ready == 0);
  probe_async_status = 0;
  assert(lf_nccl_communicator_poll_ready(communicator, &ready) == LF_NCCL_OK);
  assert(ready == 1);
  return communicator;
}

static void submit(
  lf_nccl_communicator *communicator,
  lf_device_interop_probe *device,
  uint64_t generation
) {
  assert(lf_nccl_communicator_submit_bf16(
    communicator,
    generation,
    1U,
    1U,
    0,
    1,
    lf_device_interop_probe_context(device),
    lf_device_interop_probe_send(device),
    0,
    16,
    lf_device_interop_probe_receive(device),
    0,
    16,
    lf_device_interop_probe_queue(device)
  ) == LF_NCCL_OK);
  assert(lf_device_interop_probe_active_count(device) == 4);
}

static void abort_and_free(
  lf_nccl_communicator *communicator,
  lf_device_interop_probe *device
) {
  assert(lf_nccl_communicator_abort(communicator) == LF_NCCL_OK);
  assert(lf_device_interop_probe_active_count(device) == 0);
  assert(lf_device_interop_probe_context_children(device) == 0);
  probe_finalizer(communicator);
  free(communicator);
}

static void test_pending_and_complete(lf_nccl_api *api) {
  lf_device_interop_probe device;
  lf_device_interop_probe_init(&device);
  lf_nccl_communicator *communicator = create_ready(api, &device, 1U);
  submit(communicator, &device, 1U);
  probe_async_status = LF_NCCL_IN_PROGRESS;
  assert(lunaflux_nccl_communicator_poll_collective_state(communicator) == 0);
  assert(lf_device_interop_probe_active_count(&device) == 4);
  probe_async_status = 0;
  lf_device_interop_probe_query_result(&device, 600);
  assert(lunaflux_nccl_communicator_poll_collective_state(communicator) == 0);
  assert(lf_device_interop_probe_active_count(&device) == 4);
  lf_device_interop_probe_query_result(&device, 0);
  assert(lunaflux_nccl_communicator_poll_collective_state(communicator) == 1);
  assert(lf_device_interop_probe_active_count(&device) == 0);
  abort_and_free(communicator, &device);
}

static void test_failure_retains_until_abort(lf_nccl_api *api) {
  lf_device_interop_probe device;
  lf_device_interop_probe_init(&device);
  lf_nccl_communicator *communicator = create_ready(api, &device, 2U);
  submit(communicator, &device, 2U);
  probe_async_status = 1;
  assert(
    lunaflux_nccl_communicator_poll_collective_state(communicator) ==
      LF_NCCL_RUNTIME_FAILURE
  );
  assert(lf_device_interop_probe_active_count(&device) == 4);
  assert(lf_device_interop_probe_context_close(&device) == LF_BUSY);
  abort_and_free(communicator, &device);
}

static void test_enqueue_failure_retains(lf_nccl_api *api) {
  lf_device_interop_probe device;
  lf_device_interop_probe_init(&device);
  lf_nccl_communicator *communicator = create_ready(api, &device, 3U);
  probe_submit_result = 1;
  assert(lf_nccl_communicator_submit_bf16(
    communicator,
    3U,
    1U,
    1U,
    0,
    1,
    lf_device_interop_probe_context(&device),
    lf_device_interop_probe_send(&device),
    0,
    16,
    lf_device_interop_probe_receive(&device),
    0,
    16,
    lf_device_interop_probe_queue(&device)
  ) == LF_NCCL_RUNTIME_FAILURE);
  assert(lf_device_interop_probe_active_count(&device) == 4);
  abort_and_free(communicator, &device);
  probe_submit_result = LF_NCCL_IN_PROGRESS;
}

int main(void) {
  lf_nccl_api api;
  initialize_api(&api);
  test_pending_and_complete(&api);
  test_failure_retains_until_abort(&api);
  test_enqueue_failure_retains(&api);
  assert(probe_increfs == probe_decrefs);
  return 0;
}
