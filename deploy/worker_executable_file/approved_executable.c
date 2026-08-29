#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif

#include "../../internal/approved_fs_capability/approved_fs_capability.h"

#ifndef LF_EXECUTABLE_CLOSE
#define LF_EXECUTABLE_CLOSE close
#endif
#ifndef LF_EXECUTABLE_PREAD
#define LF_EXECUTABLE_PREAD pread
#endif
#ifndef LF_EXECUTABLE_SNAPSHOT_HOOK
#define LF_EXECUTABLE_SNAPSHOT_HOOK(stage, fd) ((void)(stage), (void)(fd))
#endif
#ifndef LF_EXECUTABLE_PIN_FAILURE
#define LF_EXECUTABLE_PIN_FAILURE() 0
#endif

enum {
  LF_EXECUTABLE_OK = 0,
  LF_EXECUTABLE_INVALID = 1,
  LF_EXECUTABLE_UNAVAILABLE = 2,
  LF_EXECUTABLE_UNSUPPORTED = 3,
  LF_EXECUTABLE_CLOSED = 4,
  LF_EXECUTABLE_TRUNCATED = 5,
  LF_EXECUTABLE_FAILED = 6,
  LF_EXECUTABLE_BUSY = 7,
  LF_EXECUTABLE_CHANGED = 8,
  LF_EXECUTABLE_TOO_LARGE = 9
};

static int lf_exec_close(int fd) {
  if (fd < 0) return LF_EXECUTABLE_OK;
  int status;
#if defined(__APPLE__)
  do {
    status = LF_EXECUTABLE_CLOSE(fd);
  } while (status != 0 && errno == EINTR);
#else
  status = LF_EXECUTABLE_CLOSE(fd);
#endif
  return status == 0 ? LF_EXECUTABLE_OK : LF_EXECUTABLE_FAILED;
}

static int lf_exec_begin(lf_approved_executable *owner, int *fd) {
  if (owner == NULL || fd == NULL) return LF_EXECUTABLE_CLOSED;
  uint32_t state = atomic_load_explicit(&owner->state, memory_order_acquire);
  for (;;) {
    if ((state & LF_APPROVED_STATE_CLOSING) != 0) {
      return LF_EXECUTABLE_CLOSED;
    }
    if (state == LF_APPROVED_ACTIVE_MAX) return LF_EXECUTABLE_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &owner->state, &state, state + 1,
          memory_order_acq_rel, memory_order_acquire
        )) {
      *fd = owner->fd;
      if (*fd >= 0) return LF_EXECUTABLE_OK;
      (void)atomic_fetch_sub_explicit(&owner->state, 1, memory_order_release);
      return LF_EXECUTABLE_CLOSED;
    }
  }
}

static void lf_exec_end(lf_approved_executable *owner) {
  (void)atomic_fetch_sub_explicit(&owner->state, 1, memory_order_release);
}

static void lf_exec_finalize(void *object) {
  lf_approved_executable *owner = (lf_approved_executable *)object;
  uint32_t expected = 0;
  if (!atomic_compare_exchange_strong_explicit(
        &owner->state, &expected, LF_APPROVED_STATE_CLOSING,
        memory_order_acq_rel, memory_order_acquire
      )) return;
  int fd = owner->fd;
  owner->fd = -1;
  (void)lf_exec_close(fd);
  atomic_store_explicit(
    &owner->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
}

static lf_approved_executable *lf_exec_new(void) {
  lf_approved_executable *owner =
    (lf_approved_executable *)moonbit_make_external_object(
      lf_exec_finalize, sizeof(lf_approved_executable)
    );
  owner->fd = -1;
  owner->spawn_supported = 0;
  atomic_init(&owner->state, LF_APPROVED_STATE_CLOSED);
  return owner;
}

static int lf_exec_path_component(const uint8_t *path, int32_t start,
                                  int32_t length, char *component) {
  if (length <= 0 || length > 4096) return 0;
  int32_t dots = 0;
  for (int32_t index = 0; index < length; index++) {
    uint8_t value = path[start + index];
    int valid = (value >= 'A' && value <= 'Z') ||
      (value >= 'a' && value <= 'z') ||
      (value >= '0' && value <= '9') ||
      value == '_' || value == '-' || value == '.';
    if (!valid) return 0;
    component[index] = (char)value;
    if (value == '.') dots++;
  }
  if (length <= 2 && dots == length) return 0;
  component[length] = '\0';
  return 1;
}

static int lf_exec_open_no_follow(moonbit_bytes_t path, int32_t *status) {
  if (path == NULL || status == NULL) return -1;
  int32_t length = Moonbit_array_length(path);
  if (length < 4 || length > 4096 || path[0] != '/') {
    *status = LF_EXECUTABLE_INVALID;
    return -1;
  }
  int current = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (current < 0) {
    *status = LF_EXECUTABLE_FAILED;
    return -1;
  }
  int32_t start = 1;
  char component[4097];
  for (int32_t cursor = 1; cursor <= length; cursor++) {
    if (cursor != length && path[cursor] != '/') continue;
    int32_t component_length = cursor - start;
    if (!lf_exec_path_component(path, start, component_length, component)) {
      (void)lf_exec_close(current);
      *status = LF_EXECUTABLE_INVALID;
      return -1;
    }
    int final = cursor == length;
    int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
    if (!final) flags |= O_DIRECTORY;
    if (final) flags |= O_NONBLOCK;
    int next = openat(current, component, flags);
    int open_error = errno;
    int close_status = lf_exec_close(current);
    if (next < 0) {
      *status = open_error == ENOENT || open_error == ENOTDIR
        ? LF_EXECUTABLE_UNAVAILABLE :
        (open_error == ELOOP
          ? LF_EXECUTABLE_UNSUPPORTED : LF_EXECUTABLE_FAILED);
      return -1;
    }
    if (close_status != LF_EXECUTABLE_OK) {
      (void)lf_exec_close(next);
      *status = LF_EXECUTABLE_FAILED;
      return -1;
    }
    current = next;
    start = cursor + 1;
  }
  *status = LF_EXECUTABLE_OK;
  return current;
}

MOONBIT_FFI_EXPORT
lf_approved_executable *lunaflux_worker_executable_open(
  moonbit_bytes_t path,
  int64_t maximum_bytes,
  int32_t *status
) {
  lf_approved_executable *owner = lf_exec_new();
  if (status == NULL) return owner;
  *status = LF_EXECUTABLE_INVALID;
  if (maximum_bytes <= 0 || maximum_bytes > 1073741824LL) return owner;
  int fd = lf_exec_open_no_follow(path, status);
  if (fd < 0) return owner;
  struct stat info;
  if (fstat(fd, &info) != 0) {
    (void)lf_exec_close(fd);
    *status = LF_EXECUTABLE_FAILED;
    return owner;
  }
  if (!S_ISREG(info.st_mode) ||
      (info.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0) {
    (void)lf_exec_close(fd);
    *status = LF_EXECUTABLE_UNSUPPORTED;
    return owner;
  }
  if (info.st_size <= 0 || info.st_size > maximum_bytes) {
    (void)lf_exec_close(fd);
    *status = info.st_size > maximum_bytes
      ? LF_EXECUTABLE_TOO_LARGE : LF_EXECUTABLE_UNSUPPORTED;
    return owner;
  }
  owner->fd = fd;
#if defined(__linux__)
  owner->spawn_supported = 1;
#else
  owner->spawn_supported = 0;
#endif
  atomic_store_explicit(&owner->state, 0, memory_order_release);
  *status = LF_EXECUTABLE_OK;
  return owner;
}

#if defined(__linux__)
static int lf_exec_memfd(void) {
  return (int)syscall(SYS_memfd_create, "lunaflux-worker",
                      MFD_CLOEXEC | MFD_ALLOW_SEALING);
}

static int lf_exec_write_all(int fd, const uint8_t *bytes, int32_t length) {
  int32_t cursor = 0;
  while (cursor < length) {
    ssize_t count = write(fd, bytes + cursor, (size_t)(length - cursor));
    if (count > 0) cursor += (int32_t)count;
    else if (count < 0 && errno == EINTR) continue;
    else return 0;
  }
  return 1;
}
#endif

static moonbit_bytes_t lf_exec_snapshot_failure(moonbit_bytes_t failure) {
  if (failure == NULL) return NULL;
  moonbit_incref(failure);
  return failure;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_worker_executable_snapshot_and_pin(
  lf_approved_executable *owner,
  int64_t maximum_bytes,
  moonbit_bytes_t failure,
  int32_t *status
) {
  if (status == NULL) return lf_exec_snapshot_failure(failure);
  *status = LF_EXECUTABLE_INVALID;
  if (maximum_bytes <= 0 || maximum_bytes > INT32_MAX) {
    return lf_exec_snapshot_failure(failure);
  }
  int fd = -1;
  *status = lf_exec_begin(owner, &fd);
  if (*status != LF_EXECUTABLE_OK) {
    return lf_exec_snapshot_failure(failure);
  }
  struct stat before;
  if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode)) {
    *status = LF_EXECUTABLE_FAILED;
    goto fail_before_allocation;
  }
  if (before.st_size <= 0 || before.st_size > maximum_bytes ||
      before.st_size > INT32_MAX) {
    *status = before.st_size > maximum_bytes
      ? LF_EXECUTABLE_TOO_LARGE : LF_EXECUTABLE_UNSUPPORTED;
    goto fail_before_allocation;
  }
  int32_t length = (int32_t)before.st_size;
  moonbit_bytes_t snapshot = moonbit_make_bytes(length, 0);
  LF_EXECUTABLE_SNAPSHOT_HOOK(1, fd);
  int32_t cursor = 0;
  while (cursor < length) {
    ssize_t count = LF_EXECUTABLE_PREAD(
      fd, snapshot + cursor, (size_t)(length - cursor), cursor
    );
    if (count > 0) cursor += (int32_t)count;
    else if (count == 0) {
      *status = LF_EXECUTABLE_TRUNCATED;
      goto fail_after_allocation;
    } else if (errno != EINTR) {
      *status = LF_EXECUTABLE_FAILED;
      goto fail_after_allocation;
    }
  }
  LF_EXECUTABLE_SNAPSHOT_HOOK(2, fd);
  struct stat after;
  if (fstat(fd, &after) != 0) {
    *status = LF_EXECUTABLE_FAILED;
    goto fail_after_allocation;
  }
#if defined(__APPLE__)
#define LF_STAT_MTIME_SEC(info) ((info).st_mtimespec.tv_sec)
#define LF_STAT_MTIME_NSEC(info) ((info).st_mtimespec.tv_nsec)
#define LF_STAT_CTIME_SEC(info) ((info).st_ctimespec.tv_sec)
#define LF_STAT_CTIME_NSEC(info) ((info).st_ctimespec.tv_nsec)
#else
#define LF_STAT_MTIME_SEC(info) ((info).st_mtim.tv_sec)
#define LF_STAT_MTIME_NSEC(info) ((info).st_mtim.tv_nsec)
#define LF_STAT_CTIME_SEC(info) ((info).st_ctim.tv_sec)
#define LF_STAT_CTIME_NSEC(info) ((info).st_ctim.tv_nsec)
#endif
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      LF_STAT_MTIME_SEC(before) != LF_STAT_MTIME_SEC(after) ||
      LF_STAT_MTIME_NSEC(before) != LF_STAT_MTIME_NSEC(after) ||
      LF_STAT_CTIME_SEC(before) != LF_STAT_CTIME_SEC(after) ||
      LF_STAT_CTIME_NSEC(before) != LF_STAT_CTIME_NSEC(after)) {
    *status = LF_EXECUTABLE_CHANGED;
    goto fail_after_allocation;
  }
#if defined(__linux__)
  int sealed = lf_exec_memfd();
  if (LF_EXECUTABLE_PIN_FAILURE() || sealed < 0 ||
      !lf_exec_write_all(sealed, snapshot, length) ||
      fchmod(sealed, 0500) != 0 ||
      fcntl(sealed, F_ADD_SEALS,
            F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) != 0) {
    if (sealed >= 0) (void)lf_exec_close(sealed);
    *status = LF_EXECUTABLE_FAILED;
    goto fail_after_allocation;
  }
  int prior = owner->fd;
  if (lf_exec_close(prior) != LF_EXECUTABLE_OK) {
    owner->fd = -1;
    (void)lf_exec_close(sealed);
    *status = LF_EXECUTABLE_FAILED;
    goto fail_after_allocation;
  }
  owner->fd = sealed;
#endif
  *status = LF_EXECUTABLE_OK;
  lf_exec_end(owner);
  return snapshot;

fail_after_allocation:
  moonbit_decref(snapshot);
fail_before_allocation:
  lf_exec_end(owner);
  return lf_exec_snapshot_failure(failure);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_worker_executable_close(lf_approved_executable *owner) {
  if (owner == NULL) return LF_EXECUTABLE_FAILED;
  uint32_t state = atomic_load_explicit(&owner->state, memory_order_acquire);
  for (;;) {
    if (state == LF_APPROVED_STATE_CLOSED) return LF_EXECUTABLE_OK;
    if (state != 0) return LF_EXECUTABLE_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &owner->state, &state, LF_APPROVED_STATE_CLOSING,
          memory_order_acq_rel, memory_order_acquire
        )) break;
  }
  int fd = owner->fd;
  owner->fd = -1;
  int status = lf_exec_close(fd);
  atomic_store_explicit(
    &owner->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_worker_executable_is_closed(lf_approved_executable *owner) {
  if (owner == NULL) return 1;
  uint32_t state = atomic_load_explicit(&owner->state, memory_order_acquire);
  return (state & LF_APPROVED_STATE_CLOSING) != 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_worker_executable_spawn_supported(
  lf_approved_executable *owner
) {
  return owner != NULL && owner->spawn_supported != 0;
}
