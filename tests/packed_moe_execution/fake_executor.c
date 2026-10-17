/* Shared fixture exercises eager and immutable captured-interval owners. */
#include "../tensor_parallel_device_worker_alloc/fake_executor.c"

int32_t lunaflux_packed_capture_begin(tp_fake_executor *e) {
  if (!e || !e->live || e->execution_mode != 0 || e->phase != TP_EXECUTOR_IDLE) return -2;
  e->execution_mode = 2;
  return 0;
}
int32_t lunaflux_packed_capture_end(tp_fake_executor *e) {
  if (!e || !e->live || e->execution_mode != 2 || e->next_kernel != e->kernel_count) return -2;
  e->execution_mode = 1;
  e->phase = TP_EXECUTOR_IDLE;
  e->next_kernel = 0;
  return 0;
}
int32_t lunaflux_packed_capture_discard(tp_fake_executor *e) {
  if (!e || !e->live) return -3;
  e->execution_mode = 0;
  return 0;
}
