/* Shared fixture also models deferred submission separately from retirement. */
#include "../tensor_parallel_device_worker_alloc/fake_collective.c"

int32_t lunaflux_packed_graph_seal(tp_fake_communicator *c, uint64_t previous, int32_t sites) {
  if (!c || !c->live || c->phase != TP_COMM_LIVE || c->deferred_count != sites ||
      c->previous_plan_sequence != previous + 1 || c->next_collective_sequence != (uint64_t)sites + 1) return -2;
  c->previous_plan_sequence = previous;
  c->next_collective_sequence = 1;
  inflight = NULL;
  return 0;
}
int32_t lunaflux_packed_graph_begin(tp_fake_communicator *c, uint64_t generation, uint64_t plan, uint64_t first) {
  if (!c || !c->live || c->phase != TP_COMM_LIVE || c->deferred_count <= 0) return -4;
  if (generation != c->generation || plan != c->previous_plan_sequence + 1 || first != c->next_collective_sequence) return -7;
  c->previous_plan_sequence = plan;
  c->next_collective_sequence += c->deferred_count;
  c->phase = TP_COMM_IN_FLIGHT;
  c->submission_polls = 2;
  inflight = c;
  return 0;
}
int32_t lunaflux_packed_graph_complete(tp_fake_communicator *c) {
  if (!c || !c->live || c->phase != TP_COMM_IN_FLIGHT) return -2;
  c->phase = TP_COMM_LIVE;
  inflight = NULL;
  return 1;
}
int32_t lunaflux_packed_graph_release(tp_fake_communicator *c) {
  if (!c || !c->live || c->phase != TP_COMM_LIVE) return -4;
  c->deferred_count = 0;
  return 0;
}
