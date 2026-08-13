#ifndef LUNAFLUX_APPROVED_FS_PRIVATE_H
#define LUNAFLUX_APPROVED_FS_PRIVATE_H

_Static_assert(sizeof(off_t) >= sizeof(int64_t),
  "approved_fs requires 64-bit off_t");

#ifndef LF_APPROVED_FS_SNAPSHOT_HOOK
#define LF_APPROVED_FS_SNAPSHOT_HOOK(stage, fd) ((void)(stage), (void)(fd))
#endif

enum {
  LF_APPROVED_OK = 0,
  LF_APPROVED_INVALID = 1,
  LF_APPROVED_UNAVAILABLE = 2,
  LF_APPROVED_UNSUPPORTED = 3,
  LF_APPROVED_CLOSED = 4,
  LF_APPROVED_TRUNCATED = 5,
  LF_APPROVED_FAILED = 6,
  LF_APPROVED_BUSY = 7,
  LF_APPROVED_CHANGED = 8,
  LF_APPROVED_TOO_LARGE = 9
};

enum {
  LF_APPROVED_ROOT = 1,
  LF_APPROVED_FILE = 2,
  LF_APPROVED_PATH_MAX = 4096
};

enum {
  LF_APPROVED_STATE_CLOSING = 0x80000000u,
  LF_APPROVED_STATE_CLOSED = 0x80000001u,
  LF_APPROVED_ACTIVE_MAX = 0x7fffffffu
};

typedef struct lf_approved_handle {
  int fd;
  int kind;
  _Atomic uint32_t state;
} lf_approved_handle;

#endif
