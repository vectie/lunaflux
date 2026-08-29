#ifndef LUNAFLUX_DEVICE_COLLECTIVE_INTEROP_H
#define LUNAFLUX_DEVICE_COLLECTIVE_INTEROP_H

#include <stddef.h>
#include <stdint.h>

#define LF_DEVICE_CONTEXT_LEASE_BYTES 16U
#define LF_DEVICE_COLLECTIVE_LEASE_BYTES 64U

typedef union lf_device_context_lease {
  max_align_t alignment;
  unsigned char opaque[LF_DEVICE_CONTEXT_LEASE_BYTES];
} lf_device_context_lease;

typedef union lf_device_collective_lease {
  max_align_t alignment;
  unsigned char opaque[LF_DEVICE_COLLECTIVE_LEASE_BYTES];
} lf_device_collective_lease;

typedef struct lf_device_context_token lf_device_context_token;
typedef struct lf_device_region_token lf_device_region_token;
typedef struct lf_device_queue_token lf_device_queue_token;

enum lf_device_interop_status {
  LF_DEVICE_INTEROP_OK = 0,
  LF_DEVICE_INTEROP_INVALID_ARGUMENT = -1,
  LF_DEVICE_INTEROP_CLOSED = -2,
  LF_DEVICE_INTEROP_BUSY = -3,
  LF_DEVICE_INTEROP_RUNTIME_FAILURE = -4,
  LF_DEVICE_INTEROP_SIZE_OVERFLOW = -5
};

enum lf_device_interop_alias_rule {
  LF_DEVICE_INTEROP_EXACT_OR_DISJOINT = 1,
  LF_DEVICE_INTEROP_SLICE_OR_DISTINCT = 2
};

lf_device_context_token *lunaflux_device_interop_context_token(
  void *context_owner
);
lf_device_region_token *lunaflux_device_interop_region_token(
  void *region_owner
);
lf_device_queue_token *lunaflux_device_interop_queue_token(void *queue_owner);

int32_t lf_device_context_lease_acquire(
  lf_device_context_lease *lease,
  lf_device_context_token *context
);
void lf_device_context_lease_release(lf_device_context_lease *lease);

int32_t lf_device_collective_lease_acquire(
  lf_device_collective_lease *lease,
  const lf_device_context_lease *context_lease,
  lf_device_context_token *context,
  lf_device_region_token *send,
  int64_t send_offset,
  int64_t send_byte_count,
  lf_device_region_token *receive,
  int64_t receive_offset,
  int64_t receive_byte_count,
  lf_device_queue_token *queue,
  int64_t alignment,
  int32_t alias_rule,
  int64_t send_relative_offset
);
int32_t lf_device_collective_lease_view(
  const lf_device_collective_lease *lease,
  uintptr_t *send_address,
  uintptr_t *receive_address,
  void **queue_token
);
int32_t lf_device_collective_lease_query(
  const lf_device_collective_lease *lease,
  int32_t *completed
);
void lf_device_collective_lease_release(lf_device_collective_lease *lease);

#endif
