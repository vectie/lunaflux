#include "nccl_abi.h"

MOONBIT_FFI_EXPORT
lf_nccl_communicator *lunaflux_nccl_communicator_create(
  int32_t admitted_version,
  const uint8_t *unique_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  int32_t *status
) {
  return lf_nccl_communicator_create_with_api(
    lf_nccl_api_get(),
    admitted_version,
    unique_id,
    world_size,
    rank,
    generation,
    previous_plan_sequence,
    status
  );
}

MOONBIT_FFI_EXPORT
lf_nccl_communicator *lunaflux_nccl_communicator_create_on_context(
  int32_t admitted_version,
  const uint8_t *unique_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  lf_device_context_token *context,
  int32_t *status
) {
  return lf_nccl_communicator_create_on_context_with_api(
    lf_nccl_api_get(),
    admitted_version,
    unique_id,
    world_size,
    rank,
    generation,
    previous_plan_sequence,
    context,
    status
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_close(
  lf_nccl_communicator *communicator
) {
  return lf_nccl_communicator_close(communicator);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_abort(
  lf_nccl_communicator *communicator
) {
  return lf_nccl_communicator_abort(communicator);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_poll_ready(
  lf_nccl_communicator *communicator,
  int32_t *ready
) {
  return lf_nccl_communicator_poll_ready(communicator, ready);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_invalidate(
  lf_nccl_communicator *communicator,
  uint64_t generation
) {
  return lf_nccl_communicator_invalidate(communicator, generation);
}

MOONBIT_FFI_EXPORT
uint64_t lunaflux_nccl_communicator_generation(
  lf_nccl_communicator *communicator
) {
  return communicator == NULL ? 0 : communicator->generation;
}

MOONBIT_FFI_EXPORT
uint64_t lunaflux_nccl_communicator_next_sequence(
  lf_nccl_communicator *communicator
) {
  return communicator == NULL ? 0 : communicator->next_collective_sequence;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_world_size(
  lf_nccl_communicator *communicator
) {
  return communicator == NULL ? 0 : communicator->world_size;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_nccl_communicator_rank(lf_nccl_communicator *communicator) {
  return communicator == NULL ? -1 : communicator->rank;
}
