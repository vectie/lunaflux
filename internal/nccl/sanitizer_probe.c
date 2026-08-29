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
static int32_t probe_collective_calls = 0;

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
  assert(world_size == 2);
  assert(rank == 1);
  assert(config != NULL && config->blocking == 0);
  *handle = (void *)(uintptr_t)0x4000;
  return LF_NCCL_IN_PROGRESS;
}

static lf_nccl_result probe_async(
  lf_nccl_handle handle,
  lf_nccl_result *status
) {
  assert(handle == (void *)(uintptr_t)0x4000 && status != NULL);
  *status = 0;
  return 0;
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
  assert(send != NULL && receive != NULL && count > 0);
  assert(datatype == 9 && reduction == 0);
  assert(communicator == (void *)(uintptr_t)0x4000);
  assert(queue == (void *)(uintptr_t)0x3000);
  probe_collective_calls += 1;
  return LF_NCCL_IN_PROGRESS;
}

static lf_nccl_result probe_all_gather(
  const void *send,
  void *receive,
  size_t count,
  int32_t datatype,
  lf_nccl_handle communicator,
  lf_nccl_stream queue
) {
  return probe_all_reduce(
    send,
    receive,
    count,
    datatype,
    0,
    communicator,
    queue
  );
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
  int32_t ready = 0;
  assert(lf_nccl_communicator_poll_ready(communicator, &ready) == LF_NCCL_OK);
  assert(ready == 1);
  return communicator;
}

static void complete_one(
  lf_nccl_communicator *communicator,
  lf_device_interop_probe *device
) {
  lf_device_interop_probe_query_result(device, 0);
  int32_t completed = 0;
  assert(lf_nccl_communicator_poll_collective(
    communicator,
    &completed
  ) == LF_NCCL_OK);
  assert(completed == 1);
  assert(lf_device_interop_probe_active_count(device) == 0);
}

static void destroy_communicator(lf_nccl_communicator *communicator) {
  assert(lf_nccl_communicator_abort(communicator) == LF_NCCL_OK);
  probe_finalizer(communicator);
  free(communicator);
}

static void test_context_and_substitution(lf_nccl_api *api) {
  lf_device_interop_probe exact;
  lf_device_interop_probe foreign;
  lf_device_interop_probe_init(&exact);
  lf_device_interop_probe_init_at(
    &foreign,
    0x5000U,
    0x30000U,
    0x40000U,
    0x6000U,
    4096U
  );
  lf_nccl_communicator *communicator = create_ready(api, &exact, 1U);
  assert(lf_device_interop_probe_context_children(&exact) == 1);
  assert(lf_device_interop_probe_context_close(&exact) == LF_BUSY);
  assert(lf_nccl_communicator_submit_bf16(
    communicator,
    1U,
    1U,
    1U,
    0,
    1,
    lf_device_interop_probe_context(&foreign),
    lf_device_interop_probe_send(&foreign),
    0,
    16,
    lf_device_interop_probe_receive(&foreign),
    0,
    16,
    lf_device_interop_probe_queue(&foreign)
  ) == LF_NCCL_INVALID_ARGUMENT);
  destroy_communicator(communicator);
  assert(lf_device_interop_probe_context_children(&exact) == 0);
}

static void test_alias_geometry(lf_nccl_api *api) {
  lf_device_interop_probe device;
  lf_device_interop_probe_init(&device);
  lf_nccl_communicator *communicator = create_ready(api, &device, 2U);
  assert(lf_nccl_communicator_submit_bf16(
    communicator,
    2U,
    1U,
    1U,
    0,
    1,
    lf_device_interop_probe_context(&device),
    lf_device_interop_probe_in_place(&device),
    0,
    16,
    lf_device_interop_probe_in_place(&device),
    0,
    16,
    lf_device_interop_probe_queue(&device)
  ) == LF_NCCL_OK);
  complete_one(communicator, &device);
  assert(lf_nccl_communicator_submit_bf16(
    communicator,
    2U,
    2U,
    2U,
    1,
    2,
    lf_device_interop_probe_context(&device),
    lf_device_interop_probe_in_place(&device),
    32,
    16,
    lf_device_interop_probe_in_place(&device),
    0,
    32,
    lf_device_interop_probe_queue(&device)
  ) == LF_NCCL_OK);
  complete_one(communicator, &device);
  destroy_communicator(communicator);
}

static void test_hostile_interop(lf_device_interop_probe *device) {
  lf_device_context_lease context_lease;
  lf_device_collective_lease collective_lease;
  memset(&context_lease, 0, sizeof(context_lease));
  memset(&collective_lease, 0, sizeof(collective_lease));
  assert(lf_device_context_lease_acquire(
    &context_lease,
    lf_device_interop_probe_context(device)
  ) == LF_DEVICE_INTEROP_OK);
  assert(lf_device_collective_lease_acquire(
    &collective_lease,
    &context_lease,
    lf_device_interop_probe_context(device),
    lf_device_interop_probe_in_place(device),
    2,
    32,
    lf_device_interop_probe_in_place(device),
    0,
    32,
    lf_device_interop_probe_queue(device),
    2,
    LF_DEVICE_INTEROP_EXACT_OR_DISJOINT,
    0
  ) == LF_DEVICE_INTEROP_INVALID_ARGUMENT);
  assert(lf_device_collective_lease_acquire(
    &collective_lease,
    &context_lease,
    lf_device_interop_probe_context(device),
    lf_device_interop_probe_send(device),
    0,
    32,
    lf_device_interop_probe_receive(device),
    0,
    32,
    lf_device_interop_probe_queue(device),
    3,
    LF_DEVICE_INTEROP_EXACT_OR_DISJOINT,
    0
  ) == LF_DEVICE_INTEROP_INVALID_ARGUMENT);
  lf_device_context_lease_release(&context_lease);
}

int main(void) {
  lf_nccl_api api;
  initialize_api(&api);
  test_context_and_substitution(&api);
  test_alias_geometry(&api);
  lf_device_interop_probe hostile;
  lf_device_interop_probe_init(&hostile);
  test_hostile_interop(&hostile);
  moonbit_decref(lf_device_interop_probe_capture_context(&hostile));
  moonbit_decref(lf_device_interop_probe_capture_region(&hostile));
  moonbit_decref(lf_device_interop_probe_capture_queue(&hostile));
  assert(probe_collective_calls == 2);
  assert(probe_increfs == probe_decrefs);
  return 0;
}
