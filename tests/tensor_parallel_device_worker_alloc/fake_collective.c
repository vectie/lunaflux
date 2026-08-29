#include "moonbit.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t generation;
  uint64_t previous_plan_sequence;
  uint64_t next_collective_sequence;
  uint64_t pending_plan_sequence;
  uint64_t pending_collective_sequence;
  int32_t world_size;
  int32_t rank;
  int32_t phase;
  int32_t ready_polls;
  int32_t collective_polls;
  int32_t live;
} tp_fake_communicator;

enum {
  TP_COMM_STARTING = 0,
  TP_COMM_LIVE = 1,
  TP_COMM_IN_FLIGHT = 2,
  TP_COMM_CLOSED = 3,
  TP_COMM_FAILED = 4,
};

static uint64_t collective_submits;
static uint64_t collective_pending_polls;
static uint64_t collective_complete_polls;
static int32_t live_communicators;

static void finalize_communicator(void *raw) {
  if (((tp_fake_communicator *)raw)->live != 0) abort();
}

int32_t lunaflux_tp_alloc_fake_runtime_admit(
  int32_t minimum,
  int32_t maximum,
  int32_t *version
) {
  if (version == NULL || minimum > 21403 || maximum < 21403) return -6;
  *version = 21403;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_group_id_create(
  int32_t version,
  uint8_t *output
) {
  if (version != 21403 || output == NULL) return -2;
  for (int32_t index = 0; index < 128; index += 1) {
    output[index] = (uint8_t)(0x5aU ^ (uint8_t)index);
  }
  return 0;
}

void *lunaflux_tp_alloc_fake_communicator_create_on_context(
  int32_t version,
  const uint8_t *group_id,
  int32_t world_size,
  int32_t rank,
  uint64_t generation,
  uint64_t previous_plan_sequence,
  void *context,
  int32_t *status
) {
  tp_fake_communicator *communicator =
    (tp_fake_communicator *)moonbit_make_external_object(
      finalize_communicator,
      sizeof(tp_fake_communicator)
    );
  memset(communicator, 0, sizeof(*communicator));
  if (version != 21403 || group_id == NULL || world_size != 2 || rank < 0 ||
      rank >= world_size || generation == 0U || context == NULL) {
    *status = -2;
    return communicator;
  }
  communicator->generation = generation;
  communicator->previous_plan_sequence = previous_plan_sequence;
  communicator->next_collective_sequence = 1U;
  communicator->world_size = world_size;
  communicator->rank = rank;
  communicator->phase = TP_COMM_STARTING;
  communicator->live = 1;
  live_communicators += 1;
  *status = 0;
  return communicator;
}

int32_t lunaflux_tp_alloc_fake_communicator_poll_ready(
  tp_fake_communicator *communicator,
  int32_t *ready
) {
  if (communicator == NULL || communicator->live == 0 || ready == NULL ||
      (communicator->phase != TP_COMM_STARTING &&
       communicator->phase != TP_COMM_LIVE)) return -3;
  if (communicator->phase == TP_COMM_LIVE) {
    *ready = 1;
    return 0;
  }
  communicator->ready_polls += 1;
  if (communicator->ready_polls == 1) {
    *ready = 0;
  } else {
    communicator->phase = TP_COMM_LIVE;
    *ready = 1;
  }
  return 0;
}

int32_t lunaflux_tp_alloc_fake_communicator_submit_bf16(
  tp_fake_communicator *communicator,
  uint64_t generation,
  uint64_t plan_sequence,
  uint64_t collective_sequence,
  int32_t operation_id,
  int32_t collective_kind,
  void *context,
  void *send,
  int64_t send_offset,
  int64_t send_elements,
  void *receive,
  int64_t receive_offset,
  int64_t receive_elements,
  void *stream
) {
  if (communicator == NULL || communicator->live == 0 ||
      communicator->phase != TP_COMM_LIVE ||
      generation != communicator->generation || plan_sequence == 0U ||
      plan_sequence < communicator->previous_plan_sequence ||
      collective_sequence != communicator->next_collective_sequence ||
      operation_id < 0 || (collective_kind != 1 && collective_kind != 2) ||
      context == NULL || send == NULL || receive == NULL || stream == NULL ||
      send_offset < 0 || receive_offset < 0 || send_elements <= 0 ||
      receive_elements <= 0) return -2;
  communicator->pending_plan_sequence = plan_sequence;
  communicator->pending_collective_sequence = collective_sequence;
  communicator->collective_polls = 0;
  communicator->phase = TP_COMM_IN_FLIGHT;
  collective_submits += 1U;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_communicator_poll_collective(
  tp_fake_communicator *communicator,
  int32_t *completed
) {
  if (communicator == NULL || communicator->live == 0 || completed == NULL ||
      communicator->phase != TP_COMM_IN_FLIGHT) return -3;
  communicator->collective_polls += 1;
  if (communicator->collective_polls == 1) {
    collective_pending_polls += 1U;
    *completed = 0;
    return 0;
  }
  collective_complete_polls += 1U;
  communicator->previous_plan_sequence = communicator->pending_plan_sequence;
  communicator->next_collective_sequence =
    communicator->pending_collective_sequence + 1U;
  communicator->phase = TP_COMM_LIVE;
  *completed = 1;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_communicator_poll_collective_state(
  tp_fake_communicator *communicator
) {
  int32_t completed = 0;
  int32_t status = lunaflux_tp_alloc_fake_communicator_poll_collective(
    communicator,
    &completed
  );
  return status == 0 ? completed : status;
}

int32_t lunaflux_tp_alloc_fake_communicator_close(
  tp_fake_communicator *communicator
) {
  if (communicator == NULL) return -2;
  if (communicator->live == 0) return 0;
  if (communicator->phase != TP_COMM_LIVE) return -4;
  communicator->phase = TP_COMM_CLOSED;
  communicator->live = 0;
  live_communicators -= 1;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_communicator_abort(
  tp_fake_communicator *communicator
) {
  if (communicator == NULL) return -2;
  if (communicator->live == 0) return 0;
  communicator->phase = TP_COMM_CLOSED;
  communicator->live = 0;
  live_communicators -= 1;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_communicator_invalidate(
  tp_fake_communicator *communicator,
  uint64_t generation
) {
  if (communicator == NULL || communicator->live == 0 ||
      generation != communicator->generation) return -7;
  communicator->phase = TP_COMM_FAILED;
  return 0;
}

uint64_t lunaflux_tp_alloc_fake_communicator_generation(
  tp_fake_communicator *communicator
) {
  return communicator == NULL ? 0U : communicator->generation;
}

uint64_t lunaflux_tp_alloc_fake_communicator_next_sequence(
  tp_fake_communicator *communicator
) {
  return communicator == NULL ? 0U : communicator->next_collective_sequence;
}

int32_t lunaflux_tp_alloc_fake_communicator_world_size(
  tp_fake_communicator *communicator
) {
  return communicator == NULL ? 0 : communicator->world_size;
}

int32_t lunaflux_tp_alloc_fake_communicator_rank(
  tp_fake_communicator *communicator
) {
  return communicator == NULL ? -1 : communicator->rank;
}

void lunaflux_tp_alloc_collective_evidence_reset(void) {
  collective_submits = 0U;
  collective_pending_polls = 0U;
  collective_complete_polls = 0U;
}

uint64_t lunaflux_tp_alloc_collective_submits(void) { return collective_submits; }
uint64_t lunaflux_tp_alloc_collective_pending_polls(void) {
  return collective_pending_polls;
}
uint64_t lunaflux_tp_alloc_collective_complete_polls(void) {
  return collective_complete_polls;
}
int32_t lunaflux_tp_alloc_live_communicators(void) {
  return live_communicators;
}
