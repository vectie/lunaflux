#include "nccl_abi.h"
#include <stdlib.h>
#include <string.h>

void lf_nccl_release_graph_leases(lf_nccl_communicator *c) {
  for (int32_t i = 0; i < c->graph_lease_count; ++i)
    lf_device_collective_lease_release(&c->graph_leases[i]);
  free(c->graph_leases);
  c->graph_leases = NULL;
  c->graph_lease_count = 0;
  c->graph_sites = 0;
  c->graph_bound = 0;
}

/* Capture uses the ordinary checked submit path. Sealing retains its exact
 * stream/operand leases for the graph lifetime, but does not count capture as
 * a model step. Only this freshly initialized, exclusive communicator binds. */
MOONBIT_FFI_EXPORT int32_t lunaflux_nccl_graph_seal(
  lf_nccl_communicator *c, uint64_t previous_plan, int32_t sites
) {
  if (c == NULL || sites <= 0 || previous_plan == UINT64_MAX) return LF_NCCL_INVALID_ARGUMENT;
  if (atomic_load(&c->state) != LF_NCCL_RESOURCE_LIVE) return LF_NCCL_CLOSED;
  if (c->phase != LF_NCCL_PHASE_READY || c->graph_in_flight ||
      (c->graph_bound && c->graph_sites != sites) ||
      c->deferred_count != sites || c->last_plan_sequence != previous_plan + 1 ||
      c->next_collective_sequence != (uint64_t)sites + 1) return LF_NCCL_INVALID_ARGUMENT;
  if (sites > INT32_MAX - c->graph_lease_count) return LF_NCCL_INVALID_ARGUMENT;
  size_t total = (size_t)c->graph_lease_count + (size_t)sites;
  if (total > SIZE_MAX / sizeof(lf_device_collective_lease)) return LF_NCCL_INVALID_ARGUMENT;
  /* Startup only; move retained leases, never release them between buckets. */
  lf_device_collective_lease *bank = calloc(total, sizeof(*bank));
  if (bank == NULL) return LF_NCCL_RUNTIME_FAILURE;
  if (c->graph_lease_count) memcpy(bank, c->graph_leases, (size_t)c->graph_lease_count * sizeof(*bank));
  memcpy(bank + c->graph_lease_count, c->deferred_leases, (size_t)sites * sizeof(*bank));
  free(c->graph_leases);
  c->graph_leases = bank;
  c->graph_lease_count += sites;
  c->graph_sites = sites;
  memset(c->deferred_leases, 0, (size_t)sites * sizeof(*bank));
  c->deferred_count = 0;
  c->deferred_stream = NULL;
  c->graph_bound = 1;
  c->last_plan_sequence = previous_plan;
  c->next_collective_sequence = 1;
  c->last_operation_id = -1;
  return LF_NCCL_OK;
}

/* One checked effect interval per replay. No payload, host allocation, CUDA
 * synchronization or collective resubmission enters this boundary. */
MOONBIT_FFI_EXPORT int32_t lunaflux_nccl_graph_begin(
  lf_nccl_communicator *c, uint64_t generation, uint64_t plan, uint64_t first
) {
  if (c == NULL) return LF_NCCL_INVALID_ARGUMENT;
  if (atomic_load(&c->state) != LF_NCCL_RESOURCE_LIVE) return LF_NCCL_CLOSED;
  if (!c->graph_bound || c->graph_in_flight || c->deferred_count || c->phase != LF_NCCL_PHASE_READY)
    return LF_NCCL_BUSY;
  if (generation != c->generation || c->last_plan_sequence == UINT64_MAX ||
      plan != c->last_plan_sequence + 1 || first != c->next_collective_sequence ||
      first > UINT64_MAX - (uint64_t)c->graph_sites) return LF_NCCL_SEQUENCE_MISMATCH;
  c->last_plan_sequence = plan;
  c->next_collective_sequence += (uint64_t)c->graph_sites;
  c->graph_in_flight = 1;
  return LF_NCCL_OK;
}

/* The caller's graph completion event has already completed. Preserve NCCL's
 * asynchronous failure boundary without querying the whole CUDA stream. */
MOONBIT_FFI_EXPORT int32_t lunaflux_nccl_graph_complete(lf_nccl_communicator *c) {
  if (c == NULL) return LF_NCCL_INVALID_ARGUMENT;
  if (atomic_load(&c->state) != LF_NCCL_RESOURCE_LIVE) return LF_NCCL_CLOSED;
  if (!c->graph_bound || !c->graph_in_flight) return LF_NCCL_INVALID_ARGUMENT;
  lf_nccl_result async = LF_NCCL_IN_PROGRESS;
  lf_nccl_result result = c->api->comm_get_async_error(c->handle, &async);
  if (result == LF_NCCL_IN_PROGRESS || (result == 0 && async == LF_NCCL_IN_PROGRESS)) return 0;
  if (result != 0 || async != 0) {
    atomic_store(&c->failed, 1);
    c->phase = LF_NCCL_PHASE_FAILED;
    return LF_NCCL_RUNTIME_FAILURE;
  }
  c->graph_in_flight = 0;
  return 1;
}

/* Graph exec must be destroyed first; abort remains the consuming escape. */
MOONBIT_FFI_EXPORT int32_t lunaflux_nccl_graph_release(lf_nccl_communicator *c) {
  if (c == NULL) return LF_NCCL_INVALID_ARGUMENT;
  if (atomic_load(&c->state) != LF_NCCL_RESOURCE_LIVE) return LF_NCCL_CLOSED;
  if (!c->graph_bound || c->graph_in_flight || c->deferred_count || c->phase != LF_NCCL_PHASE_READY) return LF_NCCL_BUSY;
  lf_nccl_release_graph_leases(c);
  return LF_NCCL_OK;
}
