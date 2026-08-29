#include "nccl_abi.h"

#include <limits.h>
#include <stdint.h>

#define NCCL_SUCCESS 0
#define NCCL_IN_PROGRESS 7
#define NCCL_SUM 0
#define NCCL_BFLOAT16 9
#define LF_NCCL_ALL_REDUCE 1
#define LF_NCCL_ALL_GATHER 2
#define LF_BF16_ALIGNMENT 2

static int32_t lf_nccl_map_interop_status(int32_t status) {
  if (status == LF_DEVICE_INTEROP_OK) return LF_NCCL_OK;
  if (status == LF_DEVICE_INTEROP_CLOSED) return LF_NCCL_CLOSED;
  if (status == LF_DEVICE_INTEROP_BUSY) return LF_NCCL_BUSY;
  if (status == LF_DEVICE_INTEROP_INVALID_ARGUMENT ||
      status == LF_DEVICE_INTEROP_SIZE_OVERFLOW) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  return LF_NCCL_RUNTIME_FAILURE;
}

static int32_t lf_nccl_element_bytes(int64_t elements, size_t *bytes) {
  if (elements <= 0 || (uint64_t)elements > SIZE_MAX / 2U) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  *bytes = (size_t)elements * 2U;
  return LF_NCCL_OK;
}

static int32_t lf_nccl_validate_shape(
  lf_nccl_communicator *communicator,
  int32_t collective_kind,
  int64_t send_elements,
  int64_t receive_elements,
  size_t *send_bytes,
  size_t *receive_bytes,
  int32_t *alias_rule,
  int64_t *send_relative_offset
) {
  int32_t status = lf_nccl_element_bytes(send_elements, send_bytes);
  if (status == LF_NCCL_OK) {
    status = lf_nccl_element_bytes(receive_elements, receive_bytes);
  }
  if (status != LF_NCCL_OK) return status;
  if (collective_kind == LF_NCCL_ALL_REDUCE) {
    if (send_elements != receive_elements) return LF_NCCL_INVALID_ARGUMENT;
    *alias_rule = LF_DEVICE_INTEROP_EXACT_OR_DISJOINT;
    *send_relative_offset = 0;
    return LF_NCCL_OK;
  }
  if (collective_kind != LF_NCCL_ALL_GATHER ||
      send_elements > INT64_MAX / communicator->world_size ||
      receive_elements != send_elements * communicator->world_size ||
      (uint64_t)communicator->rank > INT64_MAX / *send_bytes) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  *alias_rule = LF_DEVICE_INTEROP_SLICE_OR_DISTINCT;
  *send_relative_offset = (int64_t)((size_t)communicator->rank * *send_bytes);
  return LF_NCCL_OK;
}

void lf_nccl_release_in_flight(lf_nccl_communicator *communicator) {
  if (communicator != NULL) {
    lf_device_collective_lease_release(&communicator->in_flight_lease);
  }
}

int32_t lf_nccl_communicator_submit_bf16(
  lf_nccl_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind,
  lf_device_context_token *context,
  lf_device_region_token *send,
  int64_t send_offset,
  int64_t send_elements,
  lf_device_region_token *receive,
  int64_t receive_offset,
  int64_t receive_elements,
  lf_device_queue_token *stream
) {
  int32_t status = lf_nccl_collective_begin(
    communicator,
    generation,
    plan_sequence,
    collective_sequence,
    operation_id,
    collective_kind
  );
  if (status != LF_NCCL_OK) return status;
  size_t send_bytes = 0;
  size_t receive_bytes = 0;
  int32_t alias_rule = 0;
  int64_t send_relative_offset = 0;
  status = lf_nccl_validate_shape(
    communicator,
    collective_kind,
    send_elements,
    receive_elements,
    &send_bytes,
    &receive_bytes,
    &alias_rule,
    &send_relative_offset
  );
  if (status == LF_NCCL_OK &&
      (communicator->api->all_reduce == NULL ||
       communicator->api->all_gather == NULL)) {
    status = LF_NCCL_UNAVAILABLE;
  }
  if (status == LF_NCCL_OK) {
    status = lf_nccl_map_interop_status(
      lf_device_collective_lease_acquire(
        &communicator->in_flight_lease,
        &communicator->context_lease,
        context,
        send,
        send_offset,
        (int64_t)send_bytes,
        receive,
        receive_offset,
        (int64_t)receive_bytes,
        stream,
        LF_BF16_ALIGNMENT,
        alias_rule,
        send_relative_offset
      )
    );
  }
  uintptr_t send_address = 0;
  uintptr_t receive_address = 0;
  void *queue_token = NULL;
  if (status == LF_NCCL_OK) {
    status = lf_nccl_map_interop_status(lf_device_collective_lease_view(
      &communicator->in_flight_lease,
      &send_address,
      &receive_address,
      &queue_token
    ));
  }
  if (status != LF_NCCL_OK) {
    lf_nccl_collective_fail(communicator);
    return status;
  }
  communicator->in_flight_plan_sequence = plan_sequence;
  communicator->in_flight_collective_sequence = collective_sequence;
  communicator->in_flight_operation_id = operation_id;
  communicator->phase = LF_NCCL_PHASE_IN_FLIGHT;
  lf_nccl_result result = collective_kind == LF_NCCL_ALL_REDUCE
    ? communicator->api->all_reduce(
        (const void *)send_address,
        (void *)receive_address,
        (size_t)send_elements,
        NCCL_BFLOAT16,
        NCCL_SUM,
        communicator->handle,
        queue_token
      )
    : communicator->api->all_gather(
        (const void *)send_address,
        (void *)receive_address,
        (size_t)send_elements,
        NCCL_BFLOAT16,
        communicator->handle,
        queue_token
      );
  if (result != NCCL_SUCCESS && result != NCCL_IN_PROGRESS) {
    lf_nccl_collective_fail(communicator);
    return LF_NCCL_RUNTIME_FAILURE;
  }
  atomic_store(&communicator->operation_lock, 0);
  return LF_NCCL_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_submit_bf16(
  lf_nccl_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind,
  lf_device_context_token *context,
  lf_device_region_token *send,
  int64_t send_offset,
  int64_t send_elements,
  lf_device_region_token *receive,
  int64_t receive_offset,
  int64_t receive_elements,
  lf_device_queue_token *stream
) {
  return lf_nccl_communicator_submit_bf16(
    communicator,
    generation,
    plan_sequence,
    collective_sequence,
    operation_id,
    collective_kind,
    context,
    send,
    send_offset,
    send_elements,
    receive,
    receive_offset,
    receive_elements,
    stream
  );
}

int32_t lf_nccl_communicator_poll_collective(
  lf_nccl_communicator *communicator,
  int32_t *completed
) {
  if (communicator == NULL || completed == NULL) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  *completed = 0;
  if (atomic_load(&communicator->state) != LF_NCCL_RESOURCE_LIVE) {
    return LF_NCCL_CLOSED;
  }
  int unlocked = 0;
  if (!atomic_compare_exchange_strong(
        &communicator->operation_lock,
        &unlocked,
        1
      )) {
    return LF_NCCL_BUSY;
  }
  if (communicator->phase == LF_NCCL_PHASE_FAILED) {
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_FAILED;
  }
  if (communicator->phase != LF_NCCL_PHASE_IN_FLIGHT) {
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_INVALID_ARGUMENT;
  }
  lf_nccl_result async_status = NCCL_IN_PROGRESS;
  lf_nccl_result result = communicator->api->comm_get_async_error(
    communicator->handle,
    &async_status
  );
  if (result == NCCL_IN_PROGRESS ||
      (result == NCCL_SUCCESS && async_status == NCCL_IN_PROGRESS)) {
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_OK;
  }
  if (result != NCCL_SUCCESS || async_status != NCCL_SUCCESS) {
    lf_nccl_collective_fail(communicator);
    return LF_NCCL_RUNTIME_FAILURE;
  }
  int32_t queue_completed = 0;
  int32_t query_status = lf_nccl_map_interop_status(
    lf_device_collective_lease_query(
      &communicator->in_flight_lease,
      &queue_completed
    )
  );
  if (query_status != LF_NCCL_OK) {
    lf_nccl_collective_fail(communicator);
    return query_status;
  }
  if (queue_completed == 0) {
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_OK;
  }
  uint64_t plan_sequence = communicator->in_flight_plan_sequence;
  uint64_t collective_sequence =
    communicator->in_flight_collective_sequence;
  int32_t operation_id = communicator->in_flight_operation_id;
  lf_nccl_release_in_flight(communicator);
  *completed = 1;
  lf_nccl_collective_commit(
    communicator,
    plan_sequence,
    collective_sequence,
    operation_id
  );
  return LF_NCCL_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_poll_collective_state(
  lf_nccl_communicator *communicator
) {
  int32_t completed = 0;
  int32_t status = lf_nccl_communicator_poll_collective(
    communicator,
    &completed
  );
  return status == LF_NCCL_OK ? completed : status;
}
