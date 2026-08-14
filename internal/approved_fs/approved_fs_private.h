#ifndef LUNAFLUX_APPROVED_FS_PRIVATE_H
#define LUNAFLUX_APPROVED_FS_PRIVATE_H

#include "../approved_fs_capability/approved_fs_capability.h"

_Static_assert(sizeof(off_t) >= sizeof(int64_t),
  "approved_fs requires 64-bit off_t");

#ifndef LF_APPROVED_FS_SNAPSHOT_HOOK
#define LF_APPROVED_FS_SNAPSHOT_HOOK(stage, fd) ((void)(stage), (void)(fd))
#endif

#ifndef LF_APPROVED_FS_CLOSE
#define LF_APPROVED_FS_CLOSE close
#endif

enum {
  LF_APPROVED_OK = LF_APPROVED_CAPABILITY_OK,
  LF_APPROVED_INVALID = 1,
  LF_APPROVED_UNAVAILABLE = 2,
  LF_APPROVED_UNSUPPORTED = 3,
  LF_APPROVED_CLOSED = LF_APPROVED_CAPABILITY_CLOSED,
  LF_APPROVED_TRUNCATED = 5,
  LF_APPROVED_FAILED = 6,
  LF_APPROVED_BUSY = LF_APPROVED_CAPABILITY_BUSY,
  LF_APPROVED_CHANGED = 8,
  LF_APPROVED_TOO_LARGE = 9,
  LF_APPROVED_IDENTITY_MISMATCH = 10
};

enum {
  LF_APPROVED_ROOT = 1,
  LF_APPROVED_FILE = 2,
  LF_APPROVED_PATH_MAX = 4096
};

typedef struct lf_approved_handle {
  int fd;
  int kind;
  _Atomic uint32_t state;
} lf_approved_handle;

int lf_close_fd(int fd);
lf_approved_handle *lf_new_handle(int kind);
int32_t lf_begin_operation(
  lf_approved_handle *handle,
  int expected_kind,
  int *fd
);
void lf_end_operation(lf_approved_handle *handle);
int32_t lf_open_status(int error);
int lf_validate_path(const uint8_t *path, int32_t length, int absolute);
int lf_traverse(
  int starting_fd,
  const uint8_t *path,
  int32_t start,
  int32_t length,
  int final_file,
  int32_t *status
);
lf_worker_approved_roots *lunaflux_approved_fs_acquire_worker_roots(
  lf_approved_handle *model_root,
  lf_approved_handle *kernel_root,
  int32_t *status
);
lf_worker_approved_roots *lunaflux_approved_fs_prepare_worker_roots(void);
int32_t lunaflux_approved_fs_acquire_prepared_worker_roots(
  lf_worker_approved_roots *roots,
  lf_approved_handle *model_root,
  lf_approved_handle *kernel_root
);
int32_t lunaflux_approved_fs_worker_roots_is_closed(
  lf_worker_approved_roots *roots
);
int32_t lunaflux_approved_fs_worker_roots_close(
  lf_worker_approved_roots *roots
);
int32_t lunaflux_approved_fs_test_close_worker_roots_while_active(
  lf_worker_approved_roots *roots
);

#endif
