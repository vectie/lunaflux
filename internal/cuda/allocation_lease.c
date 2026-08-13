#include "resource_internal.h"

#include <stdatomic.h>
#include <string.h>

static int32_t lf_close_allocation_lease(lf_allocation_lease *lease) {
  if (lease == NULL) return LF_INVALID_ARGUMENT;
  int expected = LF_RESOURCE_LIVE;
  if (!atomic_compare_exchange_strong(
        &lease->state,
        &expected,
        LF_RESOURCE_CLOSED
      )) {
    return expected == LF_RESOURCE_CLOSED ? LF_OK : LF_BUSY;
  }
  lf_allocation *allocation = lease->allocation;
  lease->allocation = NULL;
  if (allocation != NULL) {
    lf_operation_end(&allocation->active_operations);
    moonbit_decref(allocation);
  }
  return LF_OK;
}

static void lf_finalize_allocation_lease(void *object) {
  if (lf_close_allocation_lease((lf_allocation_lease *)object) != LF_OK) {
    lf_finalize_failure();
  }
}

MOONBIT_FFI_EXPORT
lf_allocation_lease *lunaflux_cuda_allocation_lease_create(
  lf_context *context,
  lf_allocation *allocation,
  int32_t *status
) {
  lf_allocation_lease *lease =
    (lf_allocation_lease *)moonbit_make_external_object(
      lf_finalize_allocation_lease,
      sizeof(lf_allocation_lease)
    );
  memset(lease, 0, sizeof(*lease));
  atomic_init(&lease->state, LF_RESOURCE_CLOSED);
  if (status == NULL) return lease;
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return lease;
  *status = allocation == NULL
    ? LF_CLOSED
    : lf_operation_begin(
        &allocation->state,
        &allocation->active_operations
      );
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return lease;
  }
  if (allocation->context != context) {
    *status = LF_INVALID_ARGUMENT;
    lf_operation_end(&allocation->active_operations);
    lf_operation_end(&context->active_operations);
    return lease;
  }
  moonbit_incref(allocation);
  lease->allocation = allocation;
  atomic_store(&lease->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return lease;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_allocation_lease_close(lf_allocation_lease *lease) {
  return lf_close_allocation_lease(lease);
}
