#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "approved_fs_private.h"
int lf_close_fd(int fd) {
  int status;
#if defined(__APPLE__)
  do status = LF_APPROVED_FS_CLOSE(fd); while (status != 0 && errno == EINTR);
#else
  status = LF_APPROVED_FS_CLOSE(fd);
#endif
  if (status == 0) return LF_APPROVED_OK;
  /* Outside Darwin, never retry an indeterminate close. */
  return LF_APPROVED_FAILED;
}
static void lf_approved_finalize(void *object) {
  lf_approved_handle *handle = (lf_approved_handle *)object;
  uint32_t expected = 0;
  if (!atomic_compare_exchange_strong_explicit(
        &handle->state,
        &expected,
        LF_APPROVED_STATE_CLOSING,
        memory_order_acq_rel,
        memory_order_acquire
      )) {
    /* Borrowed FFI calls retain the external object, so an active operation
     * cannot race finalization. A consumed handle needs no further cleanup. */
    return;
  }
  (void)lf_close_fd(handle->fd);
  handle->fd = -1;
  atomic_store_explicit(
    &handle->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
}

lf_approved_handle *lf_new_handle(int kind) {
  lf_approved_handle *handle =
    (lf_approved_handle *)moonbit_make_external_object(
      lf_approved_finalize, sizeof(lf_approved_handle)
    );
  handle->fd = -1;
  handle->kind = kind;
  atomic_init(&handle->state, LF_APPROVED_STATE_CLOSED);
  return handle;
}

int32_t lf_begin_operation(
  lf_approved_handle *handle,
  int expected_kind,
  int *fd
) {
  if (handle == NULL || handle->kind != expected_kind) {
    return LF_APPROVED_CLOSED;
  }
  uint32_t state = atomic_load_explicit(&handle->state, memory_order_acquire);
  for (;;) {
    if ((state & LF_APPROVED_STATE_CLOSING) != 0) {
      return LF_APPROVED_CLOSED;
    }
    if (state == LF_APPROVED_ACTIVE_MAX) return LF_APPROVED_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &handle->state,
          &state,
          state + 1,
          memory_order_acq_rel,
          memory_order_acquire
        )) {
      *fd = handle->fd;
      if (*fd >= 0) return LF_APPROVED_OK;
      (void)atomic_fetch_sub_explicit(
        &handle->state, 1, memory_order_release
      );
      return LF_APPROVED_CLOSED;
    }
  }
}

void lf_end_operation(lf_approved_handle *handle) {
  (void)atomic_fetch_sub_explicit(&handle->state, 1, memory_order_release);
}

int32_t lf_open_status(int error) {
  switch (error) {
    case ENOENT:
    case EACCES:
    case EPERM:
      return LF_APPROVED_UNAVAILABLE;
    case ELOOP:
    case ENOTDIR:
      return LF_APPROVED_UNSUPPORTED;
    default:
      return LF_APPROVED_FAILED;
  }
}

MOONBIT_FFI_EXPORT
lf_approved_handle *lunaflux_approved_fs_open_root(
  moonbit_bytes_t path,
  int32_t *status
) {
  lf_approved_handle *handle = lf_new_handle(LF_APPROVED_ROOT);
  int32_t length = Moonbit_array_length(path);
  *status = LF_APPROVED_INVALID;
  if (!lf_validate_path(path, length, 1)) {
    return handle;
  }
  int root = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (root < 0) {
    *status = lf_open_status(errno);
    return handle;
  }
  int fd = lf_traverse(root, path, 1, length - 1, 0, status);
  if (fd < 0) return handle;
  handle->fd = fd;
  atomic_store_explicit(&handle->state, 0, memory_order_release);
  return handle;
}

MOONBIT_FFI_EXPORT
lf_approved_handle *lunaflux_approved_fs_open_file(
  lf_approved_handle *root,
  moonbit_bytes_t locator,
  int32_t *status
) {
  lf_approved_handle *handle = lf_new_handle(LF_APPROVED_FILE);
  *status = LF_APPROVED_INVALID;
  int32_t length = Moonbit_array_length(locator);
  if (!lf_validate_path(locator, length, 0)) {
    return handle;
  }
  int root_fd = -1;
  *status = lf_begin_operation(root, LF_APPROVED_ROOT, &root_fd);
  if (*status != LF_APPROVED_OK) return handle;
  int starting_fd = fcntl(root_fd, F_DUPFD_CLOEXEC, 0);
  lf_end_operation(root);
  if (starting_fd < 0) {
    *status = LF_APPROVED_FAILED;
    return handle;
  }
  int fd = lf_traverse(starting_fd, locator, 0, length, 1, status);
  if (fd < 0) return handle;
  handle->fd = fd;
  atomic_store_explicit(&handle->state, 0, memory_order_release);
  return handle;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_close(lf_approved_handle *handle) {
  if (handle == NULL) return LF_APPROVED_FAILED;
  uint32_t state = atomic_load_explicit(&handle->state, memory_order_acquire);
  for (;;) {
    if (state == LF_APPROVED_STATE_CLOSED) return LF_APPROVED_OK;
    if (state != 0) return LF_APPROVED_BUSY;
    if (atomic_compare_exchange_weak_explicit(
          &handle->state,
          &state,
          LF_APPROVED_STATE_CLOSING,
          memory_order_acq_rel,
          memory_order_acquire
        )) {
      break;
    }
  }
  int fd = handle->fd;
  /* Consume authority before close. POSIX leaves descriptor state unspecified
   * for some failures, so exposing retry could close a reused descriptor. */
  handle->fd = -1;
  int32_t result = lf_close_fd(fd);
  atomic_store_explicit(
    &handle->state, LF_APPROVED_STATE_CLOSED, memory_order_release
  );
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_is_closed(lf_approved_handle *handle) {
  if (handle == NULL) return 1;
  uint32_t state = atomic_load_explicit(&handle->state, memory_order_acquire);
  return (state & LF_APPROVED_STATE_CLOSING) != 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_read_exact_at(
  lf_approved_handle *file,
  uint8_t *destination,
  int32_t destination_offset,
  int32_t byte_count,
  int64_t position
) {
  int32_t destination_length = Moonbit_array_length(destination);
  if (destination_offset < 0 || byte_count < 0 || position < 0 ||
      destination_offset > destination_length ||
      byte_count > destination_length - destination_offset ||
      position > INT64_MAX - (int64_t)byte_count) {
    return LF_APPROVED_INVALID;
  }
  int file_fd = -1;
  int32_t status = lf_begin_operation(file, LF_APPROVED_FILE, &file_fd);
  if (status != LF_APPROVED_OK) return status;
  int32_t cursor = 0;
  while (cursor < byte_count) {
    ssize_t count = pread(
      file_fd,
      destination + destination_offset + cursor,
      (size_t)(byte_count - cursor),
      (off_t)(position + cursor)
    );
    if (count > 0) {
      cursor += (int32_t)count;
    } else if (count == 0) {
      lf_end_operation(file);
      return LF_APPROVED_TRUNCATED;
    } else if (errno != EINTR) {
      lf_end_operation(file);
      return LF_APPROVED_FAILED;
    }
  }
  lf_end_operation(file);
  return LF_APPROVED_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_stamp(
  lf_approved_handle *file,
  int64_t *output
) {
  if (Moonbit_array_length(output) < 5) return LF_APPROVED_INVALID;
  int file_fd = -1;
  int32_t status = lf_begin_operation(file, LF_APPROVED_FILE, &file_fd);
  if (status != LF_APPROVED_OK) return status;
  struct stat info;
  if (fstat(file_fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    lf_end_operation(file);
    return LF_APPROVED_FAILED;
  }
  output[0] = (int64_t)info.st_size;
#if defined(__APPLE__)
  output[1] = (int64_t)info.st_mtimespec.tv_sec;
  output[2] = (int64_t)info.st_mtimespec.tv_nsec;
  output[3] = (int64_t)info.st_ctimespec.tv_sec;
  output[4] = (int64_t)info.st_ctimespec.tv_nsec;
#else
  output[1] = (int64_t)info.st_mtim.tv_sec;
  output[2] = (int64_t)info.st_mtim.tv_nsec;
  output[3] = (int64_t)info.st_ctim.tv_sec;
  output[4] = (int64_t)info.st_ctim.tv_nsec;
#endif
  lf_end_operation(file);
  return LF_APPROVED_OK;
}

static int lf_file_stamp_equal(
  const struct stat *left,
  const struct stat *right
) {
  if (left->st_size != right->st_size) return 0;
#if defined(__APPLE__)
  return left->st_mtimespec.tv_sec == right->st_mtimespec.tv_sec &&
    left->st_mtimespec.tv_nsec == right->st_mtimespec.tv_nsec &&
    left->st_ctimespec.tv_sec == right->st_ctimespec.tv_sec &&
    left->st_ctimespec.tv_nsec == right->st_ctimespec.tv_nsec;
#else
  return left->st_mtim.tv_sec == right->st_mtim.tv_sec &&
    left->st_mtim.tv_nsec == right->st_mtim.tv_nsec &&
    left->st_ctim.tv_sec == right->st_ctim.tv_sec &&
    left->st_ctim.tv_nsec == right->st_ctim.tv_nsec;
#endif
}

static moonbit_bytes_t lf_snapshot_failure(moonbit_bytes_t failure) {
  moonbit_incref(failure);
  return failure;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_approved_fs_read_immutable_snapshot(
  lf_approved_handle *file,
  int64_t maximum_bytes,
  moonbit_bytes_t failure,
  int32_t *status
) {
  *status = LF_APPROVED_INVALID;
  if (maximum_bytes < 0) return lf_snapshot_failure(failure);

  int file_fd = -1;
  *status = lf_begin_operation(file, LF_APPROVED_FILE, &file_fd);
  if (*status != LF_APPROVED_OK) return lf_snapshot_failure(failure);
  struct stat before;
  if (fstat(file_fd, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size < 0) {
    *status = LF_APPROVED_FAILED;
    goto fail_before_allocation;
  }
  if ((uint64_t)before.st_size > (uint64_t)maximum_bytes ||
      (uint64_t)before.st_size > INT32_MAX ||
      (uint64_t)before.st_size > SIZE_MAX) {
    *status = LF_APPROVED_TOO_LARGE;
    goto fail_before_allocation;
  }
  int32_t byte_count = (int32_t)before.st_size;
  moonbit_bytes_t output = moonbit_make_bytes(byte_count, 0);
  LF_APPROVED_FS_SNAPSHOT_HOOK(1, file_fd);
  int32_t cursor = 0;
  while (cursor < byte_count) {
    ssize_t count = pread(
      file_fd,
      output + cursor,
      (size_t)(byte_count - cursor),
      (off_t)cursor
    );
    if (count > 0) {
      cursor += (int32_t)count;
    } else if (count == 0) {
      *status = LF_APPROVED_TRUNCATED;
      goto finish;
    } else if (errno != EINTR) {
      *status = LF_APPROVED_FAILED;
      goto finish;
    }
  }
  uint8_t trailing = 0;
  ssize_t trailing_count;
  do {
    trailing_count = pread(file_fd, &trailing, 1, (off_t)before.st_size);
  } while (trailing_count < 0 && errno == EINTR);
  if (trailing_count > 0) {
    *status = LF_APPROVED_CHANGED;
    goto finish;
  }
  if (trailing_count < 0) {
    *status = LF_APPROVED_FAILED;
    goto finish;
  }
  struct stat after;
  if (fstat(file_fd, &after) != 0 || !S_ISREG(after.st_mode)) {
    *status = LF_APPROVED_FAILED;
    goto finish;
  }
  if (!lf_file_stamp_equal(&before, &after)) {
    *status = LF_APPROVED_CHANGED;
    goto finish;
  }
  *status = LF_APPROVED_OK;

finish:
  lf_end_operation(file);
  return output;

fail_before_allocation:
  lf_end_operation(file);
  return lf_snapshot_failure(failure);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_test_close_while_active(
  lf_approved_handle *handle,
  int32_t expected_kind
) {
  int fd = -1;
  int32_t status = lf_begin_operation(handle, expected_kind, &fd);
  if (status != LF_APPROVED_OK) return status;
  (void)fd;
  status = lunaflux_approved_fs_close(handle);
  lf_end_operation(handle);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_test_mkfifo(moonbit_bytes_t path) {
  int32_t length = Moonbit_array_length(path);
  if (length <= 0 || length > LF_APPROVED_PATH_MAX ||
      memchr(path, '\0', (size_t)length) != NULL) {
    return LF_APPROVED_INVALID;
  }
  return mkfifo((const char *)path, 0600) == 0
    ? LF_APPROVED_OK
    : LF_APPROVED_FAILED;
}
