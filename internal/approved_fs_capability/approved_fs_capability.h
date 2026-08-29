#ifndef LUNAFLUX_APPROVED_FS_CAPABILITY_H
#define LUNAFLUX_APPROVED_FS_CAPABILITY_H

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

enum {
  LF_APPROVED_CAPABILITY_OK = 0,
  LF_APPROVED_CAPABILITY_INVALID = 1,
  LF_APPROVED_CAPABILITY_UNSUPPORTED = 3,
  LF_APPROVED_CAPABILITY_CLOSED = 4,
  LF_APPROVED_CAPABILITY_FAILED = 6,
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

typedef struct lf_approved_executable {
  int fd;
  int spawn_supported;
  _Atomic uint32_t state;
} lf_approved_executable;

/* Header-owned so a native caller of the capability never depends on a
 * separately linked archive. The caller owns the returned close-on-exec fd. */
static inline int32_t lf_approved_executable_duplicate(
  lf_approved_executable *owner,
  int32_t *duplicate_fd
) {
  if (owner == NULL || duplicate_fd == NULL) {
    return LF_APPROVED_CAPABILITY_INVALID;
  }
  *duplicate_fd = -1;
  uint32_t state = atomic_load_explicit(&owner->state, memory_order_acquire);
  for (;;) {
    if ((state & LF_APPROVED_STATE_CLOSING) != 0) {
      return LF_APPROVED_CAPABILITY_CLOSED;
    }
    if (state == LF_APPROVED_ACTIVE_MAX) return LF_APPROVED_CAPABILITY_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &owner->state,
          &state,
          state + 1,
          memory_order_acq_rel,
          memory_order_acquire
        )) break;
  }
  int source = owner->fd;
  int supported = owner->spawn_supported;
  int duplicate = source >= 0 && supported
    ? fcntl(source, F_DUPFD_CLOEXEC, 5) : -1;
  (void)atomic_fetch_sub_explicit(&owner->state, 1, memory_order_release);
  if (source < 0) return LF_APPROVED_CAPABILITY_CLOSED;
  if (!supported) return LF_APPROVED_CAPABILITY_UNSUPPORTED;
  if (duplicate < 0) return LF_APPROVED_CAPABILITY_FAILED;
  *duplicate_fd = duplicate;
  return LF_APPROVED_CAPABILITY_OK;
}

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
