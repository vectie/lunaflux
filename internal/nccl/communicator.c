#include "nccl_abi.h"

#include <stdlib.h>
#include <string.h>

#define NCCL_SUCCESS 0

_Static_assert(sizeof(size_t) == 8, "NCCL v2 Linux ABI requires 64-bit size_t");
_Static_assert(
  offsetof(lf_nccl_config_v21400, blocking) == 16,
  "ncclConfig_v21400 blocking offset drifted"
);
_Static_assert(
  sizeof(lf_nccl_config_v21400) == 24,
  "ncclConfig_v21400 size drifted"
);

static int32_t lf_nccl_begin_consume(lf_nccl_communicator *communicator) {
  if (communicator == NULL) return LF_NCCL_INVALID_ARGUMENT;
  int unlocked = 0;
  if (!atomic_compare_exchange_strong(
        &communicator->operation_lock,
        &unlocked,
        1
      )) {
    return LF_NCCL_BUSY;
  }
  int expected = LF_NCCL_RESOURCE_LIVE;
  if (atomic_compare_exchange_strong(
        &communicator->state,
        &expected,
        LF_NCCL_RESOURCE_CLOSING
      )) {
    return LF_NCCL_OK;
  }
  atomic_store(&communicator->operation_lock, 0);
  if (expected == LF_NCCL_RESOURCE_CLOSED) return LF_NCCL_CLOSED;
  return LF_NCCL_BUSY;
}

static int32_t lf_nccl_consume(
  lf_nccl_communicator *communicator,
  int use_abort
) {
  int32_t begin = lf_nccl_begin_consume(communicator);
  if (begin == LF_NCCL_CLOSED) return LF_NCCL_OK;
  if (begin != LF_NCCL_OK) return begin;
  if (use_abort == 0 && atomic_load(&communicator->failed) != 0) {
    atomic_store(&communicator->state, LF_NCCL_RESOURCE_LIVE);
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_FAILED;
  }
  if (use_abort == 0 && communicator->phase != LF_NCCL_PHASE_READY) {
    atomic_store(&communicator->state, LF_NCCL_RESOURCE_LIVE);
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_BUSY;
  }
  lf_nccl_handle handle = communicator->handle;
  lf_nccl_result result = NCCL_SUCCESS;
  if (handle != NULL) {
    result = use_abort != 0
      ? communicator->api->comm_abort(handle)
      : communicator->api->comm_destroy(handle);
  }
  /* Nonblocking NCCL explicitly excludes Destroy/Abort from calls that may
   * return InProgress. There is no safe retry after either consuming call. */
  if (result == LF_NCCL_IN_PROGRESS) abort();
  /* NCCL specifies that a communicator must not be accessed after destroy
   * returns. Abort is likewise consuming. Never retry either native handle,
   * including when the vendor call reports failure. */
  communicator->handle = NULL;
  atomic_store(&communicator->state, LF_NCCL_RESOURCE_CLOSED);
  lf_nccl_release_in_flight(communicator);
  /* Release after publishing terminal state so a newly unblocked context
   * close cannot observe a live communicator. */
  lf_device_context_lease_release(&communicator->context_lease);
  atomic_store(&communicator->operation_lock, 0);
  return result == NCCL_SUCCESS ? LF_NCCL_OK : LF_NCCL_RUNTIME_FAILURE;
}

int32_t lf_nccl_communicator_close(lf_nccl_communicator *communicator) {
  return lf_nccl_consume(communicator, 0);
}

int32_t lf_nccl_communicator_abort(lf_nccl_communicator *communicator) {
  return lf_nccl_consume(communicator, 1);
}

static void lf_nccl_communicator_finalize(void *object) {
  lf_nccl_communicator *communicator = (lf_nccl_communicator *)object;
  int32_t result = lf_nccl_communicator_abort(communicator);
  /* A finalizer cannot return retry authority to MoonBit or prove that a
   * failed native abort released process resources. */
  if (result != LF_NCCL_OK) abort();
}

lf_nccl_communicator *lf_nccl_communicator_create_with_api(
  lf_nccl_api *api,
  int32_t admitted_version,
  const uint8_t *unique_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  int32_t *status
) {
  lf_nccl_communicator *communicator =
    (lf_nccl_communicator *)moonbit_make_external_object(
      lf_nccl_communicator_finalize,
      sizeof(lf_nccl_communicator)
    );
  memset(communicator, 0, sizeof(*communicator));
  atomic_init(&communicator->state, LF_NCCL_RESOURCE_CLOSED);
  atomic_init(&communicator->operation_lock, 0);
  atomic_init(&communicator->failed, 0);
  if (status == NULL) return communicator;
  *status = LF_NCCL_INVALID_ARGUMENT;
  if (api == NULL || unique_id == NULL ||
      admitted_version < LF_NCCL_NONBLOCKING_MINIMUM_VERSION ||
      world_size < LF_NCCL_MIN_WORLD_SIZE ||
      world_size > LF_NCCL_MAX_WORLD_SIZE || rank < 0 || rank >= world_size ||
      generation == 0 || previous_plan_sequence == UINT64_MAX) {
    return communicator;
  }
  if (api->availability != LF_NCCL_AVAILABLE) {
    *status = LF_NCCL_UNAVAILABLE;
    return communicator;
  }
  if (api->version_code != admitted_version) {
    *status = LF_NCCL_VERSION_MISMATCH;
    return communicator;
  }
  lf_nccl_unique_id id;
  memcpy(&id, unique_id, sizeof(id));
  lf_nccl_handle handle = NULL;
  lf_nccl_config_v21400 config = {
    .size = sizeof(lf_nccl_config_v21400),
    .magic = LF_NCCL_CONFIG_MAGIC,
    .version = LF_NCCL_CONFIG_VERSION,
    .blocking = 0
  };
  lf_nccl_result result = api->comm_init_rank_config(
    &handle,
    world_size,
    id,
    rank,
    &config
  );
  if (result != NCCL_SUCCESS && result != LF_NCCL_IN_PROGRESS) {
    if (handle != NULL) {
      /* Abort consumes a partially published handle. Its result cannot make
       * that handle safe to access again or return cleanup authority. */
      if (api->comm_abort(handle) != NCCL_SUCCESS) abort();
    }
    *status = LF_NCCL_RUNTIME_FAILURE;
    return communicator;
  }
  if (handle == NULL) {
    *status = LF_NCCL_RUNTIME_FAILURE;
    return communicator;
  }
  communicator->api = api;
  communicator->handle = handle;
  communicator->generation = generation;
  communicator->phase = LF_NCCL_PHASE_STARTING;
  communicator->last_plan_sequence = previous_plan_sequence;
  communicator->next_collective_sequence = 1;
  communicator->last_operation_id = -1;
  communicator->world_size = world_size;
  communicator->rank = rank;
  atomic_store(&communicator->state, LF_NCCL_RESOURCE_LIVE);
  *status = LF_NCCL_OK;
  return communicator;
}

int32_t lf_nccl_communicator_poll_ready(
  lf_nccl_communicator *communicator,
  int32_t *ready
) {
  if (communicator == NULL || ready == NULL) return LF_NCCL_INVALID_ARGUMENT;
  *ready = 0;
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
  int32_t status = LF_NCCL_OK;
  if (communicator->phase == LF_NCCL_PHASE_READY) {
    *ready = 1;
  } else if (communicator->phase == LF_NCCL_PHASE_FAILED) {
    status = LF_NCCL_FAILED;
  } else if (communicator->phase != LF_NCCL_PHASE_STARTING) {
    status = LF_NCCL_BUSY;
  } else {
    lf_nccl_result async_status = LF_NCCL_IN_PROGRESS;
    lf_nccl_result result = communicator->api->comm_get_async_error(
      communicator->handle,
      &async_status
    );
    if (result == NCCL_SUCCESS && async_status == NCCL_SUCCESS) {
      communicator->phase = LF_NCCL_PHASE_READY;
      *ready = 1;
    } else if (result != LF_NCCL_IN_PROGRESS &&
               !(result == NCCL_SUCCESS &&
                 async_status == LF_NCCL_IN_PROGRESS)) {
      atomic_store(&communicator->failed, 1);
      communicator->phase = LF_NCCL_PHASE_FAILED;
      status = LF_NCCL_RUNTIME_FAILURE;
    }
  }
  atomic_store(&communicator->operation_lock, 0);
  return status;
}

static int32_t lf_nccl_map_context_interop_status(int32_t status) {
  if (status == LF_DEVICE_INTEROP_CLOSED) return LF_NCCL_CLOSED;
  if (status == LF_DEVICE_INTEROP_BUSY) return LF_NCCL_BUSY;
  if (status == LF_DEVICE_INTEROP_INVALID_ARGUMENT ||
      status == LF_DEVICE_INTEROP_SIZE_OVERFLOW) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
  return LF_NCCL_RUNTIME_FAILURE;
}

lf_nccl_communicator *lf_nccl_communicator_create_on_context_with_api(
  lf_nccl_api *api,
  int32_t admitted_version,
  const uint8_t *unique_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  lf_device_context_token *context,
  int32_t *status
) {
  lf_device_context_lease context_lease;
  memset(&context_lease, 0, sizeof(context_lease));
  int32_t context_status = lf_device_context_lease_acquire(
    &context_lease,
    context
  );
  if (context_status != LF_DEVICE_INTEROP_OK) {
    int32_t ignored = 0;
    lf_nccl_communicator *closed = lf_nccl_communicator_create_with_api(
      api,
      admitted_version,
      unique_id,
      world_size,
      rank,
      0,
      0,
      &ignored
    );
    if (status != NULL) {
      *status = lf_nccl_map_context_interop_status(context_status);
    }
    return closed;
  }
  lf_nccl_communicator *communicator = lf_nccl_communicator_create_with_api(
    api,
    admitted_version,
    unique_id,
    world_size,
    rank,
    generation,
    previous_plan_sequence,
    status
  );
  if (atomic_load(&communicator->state) == LF_NCCL_RESOURCE_LIVE) {
    memcpy(
      &communicator->context_lease,
      &context_lease,
      sizeof(context_lease)
    );
    memset(&context_lease, 0, sizeof(context_lease));
  }
  lf_device_context_lease_release(&context_lease);
  return communicator;
}

int32_t lf_nccl_collective_begin(
  lf_nccl_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind
) {
  if (communicator == NULL) return LF_NCCL_INVALID_ARGUMENT;
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
  int32_t status = LF_NCCL_OK;
  if (atomic_load(&communicator->state) != LF_NCCL_RESOURCE_LIVE) {
    status = LF_NCCL_CLOSED;
  } else if (atomic_load(&communicator->failed) != 0) {
    status = LF_NCCL_FAILED;
  } else if (communicator->phase != LF_NCCL_PHASE_READY) {
    status = LF_NCCL_BUSY;
  } else if (generation != communicator->generation) {
    atomic_store(&communicator->failed, 1);
    communicator->phase = LF_NCCL_PHASE_FAILED;
    status = LF_NCCL_GENERATION_MISMATCH;
  } else if (plan_sequence == 0 ||
             (plan_sequence == communicator->last_plan_sequence &&
              communicator->last_operation_id < 0) ||
             (plan_sequence != communicator->last_plan_sequence &&
              (communicator->last_plan_sequence == UINT64_MAX ||
               plan_sequence != communicator->last_plan_sequence + 1)) ||
             collective_sequence == 0 ||
             collective_sequence != communicator->next_collective_sequence ||
             collective_sequence == UINT64_MAX ||
             operation_id < 0 || collective_kind < 1 || collective_kind > 3 ||
             (plan_sequence == communicator->last_plan_sequence &&
              operation_id <= communicator->last_operation_id)) {
    atomic_store(&communicator->failed, 1);
    communicator->phase = LF_NCCL_PHASE_FAILED;
    status = LF_NCCL_SEQUENCE_MISMATCH;
  } else {
    return LF_NCCL_OK;
  }
  atomic_store(&communicator->operation_lock, 0);
  return status;
}

void lf_nccl_collective_commit(
  lf_nccl_communicator *communicator,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id
) {
  if (plan_sequence > communicator->last_plan_sequence) {
    communicator->last_operation_id = -1;
  }
  communicator->last_plan_sequence = plan_sequence;
  communicator->next_collective_sequence = collective_sequence + 1;
  communicator->last_operation_id = operation_id;
  communicator->phase = LF_NCCL_PHASE_READY;
  atomic_store(&communicator->operation_lock, 0);
}

void lf_nccl_collective_fail(lf_nccl_communicator *communicator) {
  atomic_store(&communicator->failed, 1);
  communicator->phase = LF_NCCL_PHASE_FAILED;
  atomic_store(&communicator->operation_lock, 0);
}

int32_t lf_nccl_communicator_invalidate(
  lf_nccl_communicator *communicator,
  uint64_t generation
) {
  if (communicator == NULL || generation == 0) {
    return LF_NCCL_INVALID_ARGUMENT;
  }
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
  if (atomic_load(&communicator->state) != LF_NCCL_RESOURCE_LIVE) {
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_CLOSED;
  }
  if (generation != communicator->generation) {
    atomic_store(&communicator->failed, 1);
    communicator->phase = LF_NCCL_PHASE_FAILED;
    atomic_store(&communicator->operation_lock, 0);
    return LF_NCCL_GENERATION_MISMATCH;
  }
  atomic_store(&communicator->failed, 1);
  communicator->phase = LF_NCCL_PHASE_FAILED;
  atomic_store(&communicator->operation_lock, 0);
  return LF_NCCL_OK;
}
