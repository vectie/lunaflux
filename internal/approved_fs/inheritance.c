#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <fcntl.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#include "approved_fs_private.h"

static void lf_worker_roots_finalize(void *object) {
  lf_worker_approved_roots *roots = (lf_worker_approved_roots *)object;
  uint32_t expected = 0;
  if (!atomic_compare_exchange_strong_explicit(
        &roots->state, &expected, LF_APPROVED_STATE_CLOSING,
        memory_order_acq_rel, memory_order_acquire
      )) return;
  if (roots->model_fd >= 0) (void)lf_close_fd(roots->model_fd);
  if (roots->kernel_fd >= 0) (void)lf_close_fd(roots->kernel_fd);
  roots->model_fd = -1;
  roots->kernel_fd = -1;
  atomic_store_explicit(
    &roots->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
}

static lf_worker_approved_roots *lf_new_worker_roots(void) {
  lf_worker_approved_roots *roots =
    (lf_worker_approved_roots *)moonbit_make_external_object(
      lf_worker_roots_finalize, sizeof(lf_worker_approved_roots)
    );
  roots->model_fd = -1;
  roots->kernel_fd = -1;
  atomic_init(&roots->state, LF_APPROVED_STATE_CLOSED);
  return roots;
}

MOONBIT_FFI_EXPORT
lf_worker_approved_roots *lunaflux_approved_fs_acquire_worker_roots(
  lf_approved_handle *model_root,
  lf_approved_handle *kernel_root,
  int32_t *status
) {
  lf_worker_approved_roots *roots = lf_new_worker_roots();
  *status = LF_APPROVED_CLOSED;
  int model_source = -1;
  int kernel_source = -1;
  *status = lf_begin_operation(model_root, LF_APPROVED_ROOT, &model_source);
  if (*status != LF_APPROVED_OK) return roots;
  roots->model_fd = fcntl(model_source, F_DUPFD_CLOEXEC, 5);
  lf_end_operation(model_root);
  if (roots->model_fd < 0) {
    *status = LF_APPROVED_FAILED;
    return roots;
  }
  *status = lf_begin_operation(kernel_root, LF_APPROVED_ROOT, &kernel_source);
  if (*status != LF_APPROVED_OK) {
    (void)lf_close_fd(roots->model_fd);
    roots->model_fd = -1;
    return roots;
  }
  roots->kernel_fd = fcntl(kernel_source, F_DUPFD_CLOEXEC, 5);
  lf_end_operation(kernel_root);
  if (roots->kernel_fd < 0) {
    (void)lf_close_fd(roots->model_fd);
    roots->model_fd = -1;
    *status = LF_APPROVED_FAILED;
    return roots;
  }
  atomic_store_explicit(&roots->state, 0, memory_order_release);
  *status = LF_APPROVED_OK;
  return roots;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_worker_roots_is_closed(
  lf_worker_approved_roots *roots
) {
  if (roots == NULL) return 1;
  uint32_t state = atomic_load_explicit(&roots->state, memory_order_acquire);
  return (state & LF_APPROVED_STATE_CLOSING) != 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_worker_roots_close(
  lf_worker_approved_roots *roots
) {
  if (roots == NULL) return LF_APPROVED_FAILED;
  uint32_t state = atomic_load_explicit(&roots->state, memory_order_acquire);
  for (;;) {
    if (state == LF_APPROVED_STATE_CLOSED) return LF_APPROVED_OK;
    if (state != 0) return LF_APPROVED_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &roots->state, &state, LF_APPROVED_STATE_CLOSING,
          memory_order_acq_rel, memory_order_acquire
        )) break;
  }
  int model_fd = roots->model_fd;
  int kernel_fd = roots->kernel_fd;
  roots->model_fd = -1;
  roots->kernel_fd = -1;
  int32_t model_status = lf_close_fd(model_fd);
  int32_t kernel_status = lf_close_fd(kernel_fd);
  atomic_store_explicit(
    &roots->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
  return model_status == LF_APPROVED_OK && kernel_status == LF_APPROVED_OK
    ? LF_APPROVED_OK : LF_APPROVED_FAILED;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_test_close_worker_roots_while_active(
  lf_worker_approved_roots *roots
) {
  int32_t model_fd = -1;
  int32_t kernel_fd = -1;
  int32_t status = lf_worker_roots_begin(roots, &model_fd, &kernel_fd);
  if (status != LF_APPROVED_OK) return status;
  status = lunaflux_approved_fs_worker_roots_close(roots);
  lf_worker_roots_end(roots);
  return status;
}

MOONBIT_FFI_EXPORT
lf_approved_handle *lunaflux_approved_fs_open_inherited_root(
  int32_t role,
  int32_t *status
) {
  lf_approved_handle *root = lf_new_handle(LF_APPROVED_ROOT);
  *status = LF_APPROVED_INVALID;
  if (role != 0 && role != 1) return root;
  int inherited_fd = role == 0 ? 3 : 4;
  struct stat info;
  if (fstat(inherited_fd, &info) != 0) {
    (void)lf_close_fd(inherited_fd);
    *status = LF_APPROVED_UNAVAILABLE;
    return root;
  }
  if (!S_ISDIR(info.st_mode)) {
    (void)lf_close_fd(inherited_fd);
    *status = LF_APPROVED_UNSUPPORTED;
    return root;
  }
  int owned_fd = fcntl(inherited_fd, F_DUPFD_CLOEXEC, 5);
  int32_t close_status = lf_close_fd(inherited_fd);
  if (owned_fd < 0 || close_status != LF_APPROVED_OK) {
    if (owned_fd >= 0) (void)lf_close_fd(owned_fd);
    *status = LF_APPROVED_FAILED;
    return root;
  }
  root->fd = owned_fd;
  atomic_store_explicit(&root->state, 0, memory_order_release);
  *status = LF_APPROVED_OK;
  return root;
}
