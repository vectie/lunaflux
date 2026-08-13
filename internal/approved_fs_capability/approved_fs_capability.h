#ifndef LUNAFLUX_APPROVED_FS_CAPABILITY_H
#define LUNAFLUX_APPROVED_FS_CAPABILITY_H

#include <stdint.h>
#include <stdatomic.h>

enum {
  LF_APPROVED_CAPABILITY_OK = 0,
  LF_APPROVED_CAPABILITY_CLOSED = 4,
  LF_APPROVED_CAPABILITY_BUSY = 7,
  LF_APPROVED_STATE_CLOSING = 0x80000000u,
  LF_APPROVED_STATE_CLOSED = 0x80000001u,
  LF_APPROVED_STATE_PREPARING = 0x80000002u,
  LF_APPROVED_STATE_PREPARED = 0x80000003u,
  LF_APPROVED_ACTIVE_MAX = 0x7fffffffu
};

typedef struct lf_worker_approved_roots {
  int model_fd;
  int kernel_fd;
  _Atomic uint32_t state;
} lf_worker_approved_roots;

static inline int32_t lf_worker_roots_begin(
  lf_worker_approved_roots *roots,
  int32_t *model_fd,
  int32_t *kernel_fd
) {
  if (roots == NULL || model_fd == NULL || kernel_fd == NULL) {
    return LF_APPROVED_CAPABILITY_CLOSED;
  }
  uint32_t state = atomic_load_explicit(&roots->state, memory_order_acquire);
  for (;;) {
    if ((state & LF_APPROVED_STATE_CLOSING) != 0) {
      return LF_APPROVED_CAPABILITY_CLOSED;
    }
    if (state == LF_APPROVED_ACTIVE_MAX) return LF_APPROVED_CAPABILITY_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &roots->state,
          &state,
          state + 1,
          memory_order_acq_rel,
          memory_order_acquire
        )) {
      *model_fd = roots->model_fd;
      *kernel_fd = roots->kernel_fd;
      if (*model_fd >= 5 && *kernel_fd >= 5 && *model_fd != *kernel_fd) {
        return LF_APPROVED_CAPABILITY_OK;
      }
      (void)atomic_fetch_sub_explicit(
        &roots->state, 1, memory_order_release
      );
      return LF_APPROVED_CAPABILITY_CLOSED;
    }
  }
}

static inline void lf_worker_roots_end(lf_worker_approved_roots *roots) {
  (void)atomic_fetch_sub_explicit(&roots->state, 1, memory_order_release);
}

#endif
