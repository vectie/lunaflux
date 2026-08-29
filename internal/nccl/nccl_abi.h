#ifndef LUNAFLUX_NCCL_ABI_H
#define LUNAFLUX_NCCL_ABI_H

#include "../cuda/collective_interop.h"

#include <moonbit.h>

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__linux__)
#include <dlfcn.h>
#include <pthread.h>
typedef void *lf_nccl_library;
typedef pthread_once_t lf_nccl_once;
#define LF_NCCL_ONCE_INIT PTHREAD_ONCE_INIT
#else
typedef void *lf_nccl_library;
#endif

#define LF_NCCL_UNIQUE_ID_BYTES 128
#define LF_NCCL_MIN_WORLD_SIZE 2
#define LF_NCCL_MAX_WORLD_SIZE 16
#define LF_NCCL_NONBLOCKING_MINIMUM_VERSION 21403
#define LF_NCCL_CONFIG_VERSION 21403U
#define LF_NCCL_CONFIG_MAGIC 0xcafebeefU
#define LF_NCCL_IN_PROGRESS 7

typedef struct lf_nccl_config_v21400 {
  size_t size;
  uint32_t magic;
  uint32_t version;
  int32_t blocking;
} lf_nccl_config_v21400;

typedef int32_t lf_nccl_result;
typedef void *lf_nccl_handle;
typedef void *lf_nccl_stream;
typedef struct {
  char internal[LF_NCCL_UNIQUE_ID_BYTES];
} lf_nccl_unique_id;

enum lf_nccl_availability {
  LF_NCCL_AVAILABLE = 0,
  LF_NCCL_UNSUPPORTED_PLATFORM = 1,
  LF_NCCL_LIBRARY_MISSING = 2,
  LF_NCCL_ABI_INCOMPLETE = 3,
  LF_NCCL_VERSION_UNAVAILABLE = 4
};

enum lf_nccl_status {
  LF_NCCL_OK = 0,
  LF_NCCL_UNAVAILABLE = -1,
  LF_NCCL_INVALID_ARGUMENT = -2,
  LF_NCCL_CLOSED = -3,
  LF_NCCL_BUSY = -4,
  LF_NCCL_RUNTIME_FAILURE = -5,
  LF_NCCL_VERSION_MISMATCH = -6,
  LF_NCCL_GENERATION_MISMATCH = -7,
  LF_NCCL_SEQUENCE_MISMATCH = -8,
  LF_NCCL_FAILED = -9
};

enum lf_nccl_resource_state {
  LF_NCCL_RESOURCE_LIVE = 1,
  LF_NCCL_RESOURCE_CLOSING = 2,
  LF_NCCL_RESOURCE_CLOSED = 3
};

enum lf_nccl_phase {
  LF_NCCL_PHASE_STARTING = 1,
  LF_NCCL_PHASE_READY = 2,
  LF_NCCL_PHASE_IN_FLIGHT = 3,
  LF_NCCL_PHASE_FAILED = 4
};

typedef struct lf_nccl_api {
  lf_nccl_library library;
  int32_t availability;
  int32_t version_code;
  lf_nccl_result (*get_version)(int32_t *);
  lf_nccl_result (*get_unique_id)(lf_nccl_unique_id *);
  lf_nccl_result (*comm_init_rank_config)(
    lf_nccl_handle *,
    int32_t,
    lf_nccl_unique_id,
    int32_t,
    lf_nccl_config_v21400 *
  );
  lf_nccl_result (*comm_get_async_error)(lf_nccl_handle, lf_nccl_result *);
  lf_nccl_result (*comm_destroy)(lf_nccl_handle);
  lf_nccl_result (*comm_abort)(lf_nccl_handle);
  lf_nccl_result (*all_reduce)(
    const void *,
    void *,
    size_t,
    int32_t,
    int32_t,
    lf_nccl_handle,
    lf_nccl_stream
  );
  lf_nccl_result (*all_gather)(
    const void *,
    void *,
    size_t,
    int32_t,
    lf_nccl_handle,
    lf_nccl_stream
  );
} lf_nccl_api;

typedef struct lf_nccl_communicator {
  lf_nccl_api *api;
  lf_nccl_handle handle;
  lf_device_context_lease context_lease;
  lf_device_collective_lease in_flight_lease;
  atomic_int state;
  atomic_int operation_lock;
  atomic_int failed;
  int32_t phase;
  uint64_t generation;
  uint64_t last_plan_sequence;
  uint64_t next_collective_sequence;
  int32_t last_operation_id;
  int32_t world_size;
  int32_t rank;
  uint64_t in_flight_plan_sequence;
  uint64_t in_flight_collective_sequence;
  int32_t in_flight_operation_id;
} lf_nccl_communicator;

lf_nccl_api *lf_nccl_api_get(void);
int32_t lf_nccl_runtime_admit_with_api(
  lf_nccl_api *api,
  int32_t minimum_version,
  int32_t maximum_version,
  int32_t *version
);
int32_t lf_nccl_unique_id_create_with_api(
  lf_nccl_api *api,
  int32_t admitted_version,
  uint8_t *output
);
lf_nccl_communicator *lf_nccl_communicator_create_with_api(
  lf_nccl_api *api,
  int32_t admitted_version,
  const uint8_t *unique_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  int32_t *status
);
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
);
int32_t lf_nccl_communicator_close(lf_nccl_communicator *communicator);
int32_t lf_nccl_communicator_abort(lf_nccl_communicator *communicator);
int32_t lf_nccl_communicator_poll_ready(
  lf_nccl_communicator *communicator,
  int32_t *ready
);
int32_t lf_nccl_communicator_invalidate(
  lf_nccl_communicator *communicator,
  uint64_t generation
);
int32_t lf_nccl_collective_begin(
  lf_nccl_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind
);
void lf_nccl_collective_commit(
  lf_nccl_communicator *communicator,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id
);
void lf_nccl_collective_fail(lf_nccl_communicator *communicator);
void lf_nccl_release_in_flight(lf_nccl_communicator *communicator);
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
);
int32_t lf_nccl_communicator_poll_collective(
  lf_nccl_communicator *communicator,
  int32_t *completed
);

#endif
