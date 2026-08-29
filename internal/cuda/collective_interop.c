#include "collective_interop.h"
#include "resource_internal.h"

#include <stdatomic.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_READY 600

typedef struct {
  lf_context *context;
  int32_t live;
} lf_context_lease_private;

typedef struct {
  lf_context *context;
  lf_allocation *send;
  lf_allocation *receive;
  lf_child *queue;
  uintptr_t send_address;
  uintptr_t receive_address;
  int32_t receive_distinct;
  int32_t live;
} lf_collective_lease_private;

_Static_assert(
  sizeof(lf_context_lease_private) <= LF_DEVICE_CONTEXT_LEASE_BYTES,
  "device context lease storage is too small"
);
_Static_assert(
  sizeof(lf_collective_lease_private) <= LF_DEVICE_COLLECTIVE_LEASE_BYTES,
  "device collective lease storage is too small"
);
_Static_assert(
  _Alignof(lf_device_context_lease) >= _Alignof(lf_context_lease_private),
  "device context lease alignment is too small"
);
_Static_assert(
  _Alignof(lf_device_collective_lease) >=
    _Alignof(lf_collective_lease_private),
  "device collective lease alignment is too small"
);

static lf_context_lease_private *lf_context_lease_private_mut(
  lf_device_context_lease *lease
) {
  return (lf_context_lease_private *)(void *)lease->opaque;
}

static const lf_context_lease_private *lf_context_lease_private_const(
  const lf_device_context_lease *lease
) {
  return (const lf_context_lease_private *)(const void *)lease->opaque;
}

static lf_collective_lease_private *lf_collective_lease_private_mut(
  lf_device_collective_lease *lease
) {
  return (lf_collective_lease_private *)(void *)lease->opaque;
}

static const lf_collective_lease_private *lf_collective_lease_private_const(
  const lf_device_collective_lease *lease
) {
  return (const lf_collective_lease_private *)(const void *)lease->opaque;
}

static int32_t lf_interop_map_status(int32_t status) {
  if (status == LF_OK) return LF_DEVICE_INTEROP_OK;
  if (status == LF_CLOSED) return LF_DEVICE_INTEROP_CLOSED;
  if (status == LF_BUSY) return LF_DEVICE_INTEROP_BUSY;
  if (status == LF_SIZE_OVERFLOW) return LF_DEVICE_INTEROP_SIZE_OVERFLOW;
  if (status == LF_INVALID_ARGUMENT) {
    return LF_DEVICE_INTEROP_INVALID_ARGUMENT;
  }
  return LF_DEVICE_INTEROP_RUNTIME_FAILURE;
}

static void *lf_retain_token(void *owner) {
  if (owner != NULL) moonbit_incref(owner);
  return owner;
}

MOONBIT_FFI_EXPORT
void lunaflux_cuda_collective_interop_link_anchor(void) {}

MOONBIT_FFI_EXPORT
lf_device_context_token *lunaflux_device_interop_context_token(
  void *context_owner
) {
  return (lf_device_context_token *)lf_retain_token(context_owner);
}

MOONBIT_FFI_EXPORT
lf_device_region_token *lunaflux_device_interop_region_token(
  void *region_owner
) {
  return (lf_device_region_token *)lf_retain_token(region_owner);
}

MOONBIT_FFI_EXPORT
lf_device_queue_token *lunaflux_device_interop_queue_token(void *queue_owner) {
  return (lf_device_queue_token *)lf_retain_token(queue_owner);
}

int32_t lf_device_context_lease_acquire(
  lf_device_context_lease *lease,
  lf_device_context_token *token
) {
  if (lease == NULL || token == NULL) {
    return LF_DEVICE_INTEROP_INVALID_ARGUMENT;
  }
  lf_context_lease_private *private_ = lf_context_lease_private_mut(lease);
  if (private_->live != 0) return LF_DEVICE_INTEROP_BUSY;
  lf_context *context = (lf_context *)(void *)token;
  int32_t status = lf_operation_begin(
    &context->state,
    &context->active_operations
  );
  int acquired = status == LF_OK;
  if (status == LF_OK) status = lf_context_current(context);
  if (status != LF_OK) {
    if (acquired != 0) lf_operation_end(&context->active_operations);
    return lf_interop_map_status(status);
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  private_->context = context;
  private_->live = 1;
  lf_operation_end(&context->active_operations);
  return LF_DEVICE_INTEROP_OK;
}

void lf_device_context_lease_release(lf_device_context_lease *lease) {
  if (lease == NULL) return;
  lf_context_lease_private *private_ = lf_context_lease_private_mut(lease);
  if (private_->live == 0) return;
  lf_context *context = private_->context;
  memset(private_, 0, sizeof(*private_));
  lf_release_context_child(context);
}

static int lf_regions_disjoint(
  int64_t left_offset,
  size_t left_bytes,
  int64_t right_offset,
  size_t right_bytes
) {
  uint64_t left = (uint64_t)left_offset;
  uint64_t right = (uint64_t)right_offset;
  return left <= right
    ? right - left >= left_bytes
    : left - right >= right_bytes;
}

static int32_t lf_validate_alias(
  lf_allocation *send,
  int64_t send_offset,
  size_t send_bytes,
  lf_allocation *receive,
  int64_t receive_offset,
  size_t receive_bytes,
  int32_t alias_rule,
  int64_t send_relative_offset
) {
  if (send != receive) return LF_OK;
  if (alias_rule == LF_DEVICE_INTEROP_EXACT_OR_DISJOINT) {
    if (send_offset == receive_offset && send_bytes == receive_bytes) {
      return LF_OK;
    }
    return lf_regions_disjoint(
      send_offset,
      send_bytes,
      receive_offset,
      receive_bytes
    ) ? LF_OK : LF_INVALID_ARGUMENT;
  }
  if (alias_rule != LF_DEVICE_INTEROP_SLICE_OR_DISTINCT ||
      send_relative_offset < 0 || send_bytes > receive_bytes ||
      send_relative_offset > INT64_MAX - receive_offset ||
      send_offset != receive_offset + send_relative_offset ||
      (uint64_t)send_relative_offset > receive_bytes - send_bytes) {
    return LF_INVALID_ARGUMENT;
  }
  return LF_OK;
}

static void lf_release_acquired(
  lf_context *context,
  lf_allocation *send,
  lf_allocation *receive,
  lf_child *queue,
  int context_acquired,
  int send_acquired,
  int receive_acquired,
  int queue_acquired
) {
  if (queue_acquired != 0) lf_operation_end(&queue->active_operations);
  if (receive_acquired != 0) {
    lf_operation_end(&receive->active_operations);
  }
  if (send_acquired != 0) lf_operation_end(&send->active_operations);
  if (context_acquired != 0) lf_operation_end(&context->active_operations);
}

int32_t lf_device_collective_lease_acquire(
  lf_device_collective_lease *lease,
  const lf_device_context_lease *context_lease,
  lf_device_context_token *context_token,
  lf_device_region_token *send_token,
  int64_t send_offset,
  int64_t send_byte_count,
  lf_device_region_token *receive_token,
  int64_t receive_offset,
  int64_t receive_byte_count,
  lf_device_queue_token *queue_token,
  int64_t alignment,
  int32_t alias_rule,
  int64_t send_relative_offset
) {
  if (lease == NULL || context_lease == NULL || context_token == NULL ||
      send_token == NULL || receive_token == NULL || queue_token == NULL ||
      send_offset < 0 || receive_offset < 0 || send_byte_count <= 0 ||
      receive_byte_count <= 0 || alignment <= 0 ||
      (uint64_t)send_byte_count > SIZE_MAX ||
      (uint64_t)receive_byte_count > SIZE_MAX ||
      (uint64_t)alignment > SIZE_MAX) {
    return LF_DEVICE_INTEROP_INVALID_ARGUMENT;
  }
  lf_collective_lease_private *private_ =
    lf_collective_lease_private_mut(lease);
  const lf_context_lease_private *context_private =
    lf_context_lease_private_const(context_lease);
  if (private_->live != 0) return LF_DEVICE_INTEROP_BUSY;
  if (context_private->live == 0) return LF_DEVICE_INTEROP_CLOSED;
  lf_context *context = (lf_context *)(void *)context_token;
  lf_allocation *send = (lf_allocation *)(void *)send_token;
  lf_allocation *receive = (lf_allocation *)(void *)receive_token;
  lf_child *queue = (lf_child *)(void *)queue_token;
  int context_acquired = 0;
  int send_acquired = 0;
  int receive_acquired = 0;
  int queue_acquired = 0;
  int32_t status = context == context_private->context
    ? lf_operation_begin(&context->state, &context->active_operations)
    : LF_INVALID_ARGUMENT;
  context_acquired = status == LF_OK;
  if (status == LF_OK) {
    status = lf_operation_begin(&send->state, &send->active_operations);
    send_acquired = status == LF_OK;
  }
  if (status == LF_OK && receive != send) {
    status = lf_operation_begin(
      &receive->state,
      &receive->active_operations
    );
    receive_acquired = status == LF_OK;
  }
  if (status == LF_OK) {
    status = lf_operation_begin(&queue->state, &queue->active_operations);
    queue_acquired = status == LF_OK;
  }
  if (status == LF_OK &&
      (send->context != context || receive->context != context ||
       queue->context != context)) {
    status = LF_INVALID_ARGUMENT;
  }
  size_t send_bytes = (size_t)send_byte_count;
  size_t receive_bytes = (size_t)receive_byte_count;
  CUdeviceptr send_address = 0;
  CUdeviceptr receive_address = 0;
  if (status == LF_OK) {
    status = lf_validate_alias(
      send,
      send_offset,
      send_bytes,
      receive,
      receive_offset,
      receive_bytes,
      alias_rule,
      send_relative_offset
    );
  }
  if (status == LF_OK) {
    status = lf_allocation_region_address(
      send,
      send_offset,
      send_bytes,
      (size_t)alignment,
      0,
      &send_address
    );
  }
  if (status == LF_OK) {
    status = lf_allocation_region_address(
      receive,
      receive_offset,
      receive_bytes,
      (size_t)alignment,
      0,
      &receive_address
    );
  }
  if (status == LF_OK) status = lf_context_current(context);
  if (status != LF_OK) {
    lf_release_acquired(
      context,
      send,
      receive,
      queue,
      context_acquired,
      send_acquired,
      receive_acquired,
      queue_acquired
    );
    return lf_interop_map_status(status);
  }
  moonbit_incref(send);
  if (receive != send) moonbit_incref(receive);
  moonbit_incref(queue);
  private_->context = context;
  private_->send = send;
  private_->receive = receive;
  private_->queue = queue;
  private_->send_address = (uintptr_t)send_address;
  private_->receive_address = (uintptr_t)receive_address;
  private_->receive_distinct = receive != send;
  private_->live = 1;
  return LF_DEVICE_INTEROP_OK;
}

int32_t lf_device_collective_lease_view(
  const lf_device_collective_lease *lease,
  uintptr_t *send_address,
  uintptr_t *receive_address,
  void **queue_token
) {
  if (lease == NULL || send_address == NULL || receive_address == NULL ||
      queue_token == NULL) {
    return LF_DEVICE_INTEROP_INVALID_ARGUMENT;
  }
  const lf_collective_lease_private *private_ =
    lf_collective_lease_private_const(lease);
  if (private_->live == 0) return LF_DEVICE_INTEROP_CLOSED;
  *send_address = private_->send_address;
  *receive_address = private_->receive_address;
  *queue_token = private_->queue->handle;
  return LF_DEVICE_INTEROP_OK;
}

int32_t lf_device_collective_lease_query(
  const lf_device_collective_lease *lease,
  int32_t *completed
) {
  if (lease == NULL || completed == NULL) {
    return LF_DEVICE_INTEROP_INVALID_ARGUMENT;
  }
  *completed = 0;
  const lf_collective_lease_private *private_ =
    lf_collective_lease_private_const(lease);
  if (private_->live == 0) return LF_DEVICE_INTEROP_CLOSED;
  int32_t status = lf_context_current(private_->context);
  if (status != LF_OK) return lf_interop_map_status(status);
  CUresult result = private_->context->api->cuStreamQuery(
    (CUstream)private_->queue->handle
  );
  if (result == CUDA_ERROR_NOT_READY) return LF_DEVICE_INTEROP_OK;
  if (result != CUDA_SUCCESS) return LF_DEVICE_INTEROP_RUNTIME_FAILURE;
  *completed = 1;
  return LF_DEVICE_INTEROP_OK;
}

void lf_device_collective_lease_release(lf_device_collective_lease *lease) {
  if (lease == NULL) return;
  lf_collective_lease_private *private_ =
    lf_collective_lease_private_mut(lease);
  if (private_->live == 0) return;
  lf_allocation *send = private_->send;
  lf_allocation *receive = private_->receive;
  lf_child *queue = private_->queue;
  lf_context *context = private_->context;
  int receive_distinct = private_->receive_distinct;
  memset(private_, 0, sizeof(*private_));
  lf_operation_end(&queue->active_operations);
  if (receive_distinct != 0) {
    lf_operation_end(&receive->active_operations);
  }
  lf_operation_end(&send->active_operations);
  lf_operation_end(&context->active_operations);
  moonbit_decref(queue);
  if (receive_distinct != 0) moonbit_decref(receive);
  moonbit_decref(send);
}
