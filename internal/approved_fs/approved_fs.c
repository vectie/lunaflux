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

_Static_assert(sizeof(off_t) >= sizeof(int64_t), "approved_fs requires 64-bit off_t");

enum {
  LF_APPROVED_OK = 0,
  LF_APPROVED_INVALID = 1,
  LF_APPROVED_UNAVAILABLE = 2,
  LF_APPROVED_UNSUPPORTED = 3,
  LF_APPROVED_CLOSED = 4,
  LF_APPROVED_TRUNCATED = 5,
  LF_APPROVED_FAILED = 6,
  LF_APPROVED_BUSY = 7
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

static int lf_close_fd(int fd) {
  if (close(fd) == 0) return LF_APPROVED_OK;
  /* Never retry close. POSIX does not provide portable retry authority and a
   * reused descriptor number could otherwise close an unrelated resource. */
  if (errno == EINTR) return LF_APPROVED_OK;
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

static lf_approved_handle *lf_new_handle(int kind) {
  lf_approved_handle *handle =
    (lf_approved_handle *)moonbit_make_external_object(
      lf_approved_finalize, sizeof(lf_approved_handle)
    );
  handle->fd = -1;
  handle->kind = kind;
  atomic_init(&handle->state, LF_APPROVED_STATE_CLOSED);
  return handle;
}

static int32_t lf_begin_operation(
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

static void lf_end_operation(lf_approved_handle *handle) {
  (void)atomic_fetch_sub_explicit(&handle->state, 1, memory_order_release);
}

static int32_t lf_open_status(int error) {
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

static int lf_path_character(uint8_t value) {
  return (value >= 'A' && value <= 'Z') ||
    (value >= 'a' && value <= 'z') ||
    (value >= '0' && value <= '9') ||
    value == '_' || value == '-' || value == '.';
}

static int lf_validate_path(
  const uint8_t *path,
  int32_t length,
  int absolute
) {
  if (path == NULL || length <= 0 || length > LF_APPROVED_PATH_MAX) return 0;
  int32_t start = 0;
  if (absolute) {
    if (length < 2 || path[0] != '/') return 0;
    start = 1;
  } else if (path[0] == '/') {
    return 0;
  }
  int32_t segment_length = 0;
  int32_t segment_dots = 0;
  for (int32_t index = start; index < length; index++) {
    uint8_t value = path[index];
    if (value == '/') {
      if (segment_length == 0 ||
          (segment_length <= 2 && segment_dots == segment_length)) {
        return 0;
      }
      segment_length = 0;
      segment_dots = 0;
      continue;
    }
    if (!lf_path_character(value)) return 0;
    segment_length++;
    if (value == '.') segment_dots++;
  }
  return segment_length > 0 &&
    !(segment_length <= 2 && segment_dots == segment_length);
}

static int lf_open_component(
  int parent,
  const uint8_t *path,
  int32_t start,
  int32_t length,
  int final_file,
  int32_t *status
) {
  if (length <= 0 || length > LF_APPROVED_PATH_MAX) {
    *status = LF_APPROVED_INVALID;
    return -1;
  }
  char component[LF_APPROVED_PATH_MAX + 1];
  memcpy(component, path + start, (size_t)length);
  component[length] = '\0';
  int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  if (final_file) flags |= O_NONBLOCK;
  if (!final_file) flags |= O_DIRECTORY;
  int fd = openat(parent, component, flags);
  if (fd < 0) {
    *status = lf_open_status(errno);
    return -1;
  }
  struct stat info;
  if (fstat(fd, &info) != 0) {
    (void)lf_close_fd(fd);
    *status = LF_APPROVED_FAILED;
    return -1;
  }
  int right_type = final_file ? S_ISREG(info.st_mode) : S_ISDIR(info.st_mode);
  if (!right_type) {
    (void)lf_close_fd(fd);
    *status = LF_APPROVED_UNSUPPORTED;
    return -1;
  }
  return fd;
}

static int lf_traverse(
  int starting_fd,
  const uint8_t *path,
  int32_t start,
  int32_t length,
  int final_file,
  int32_t *status
) {
  int current = starting_fd;
  int32_t component_start = start;
  int32_t end = start + length;
  for (int32_t cursor = start; cursor <= end; cursor++) {
    int boundary = cursor == end || path[cursor] == '/';
    if (!boundary) continue;
    int is_final = cursor == end;
    int next = lf_open_component(
      current,
      path,
      component_start,
      cursor - component_start,
      final_file && is_final,
      status
    );
    (void)lf_close_fd(current);
    if (next < 0) return -1;
    current = next;
    component_start = cursor + 1;
  }
  *status = LF_APPROVED_OK;
  return current;
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
