#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "approved_fs_private.h"

static int lf_path_character(uint8_t value) {
  return (value >= 'A' && value <= 'Z') ||
    (value >= 'a' && value <= 'z') ||
    (value >= '0' && value <= '9') ||
    value == '_' || value == '-' || value == '.';
}

int lf_validate_path(
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

int lf_traverse(
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
    int32_t close_status = lf_close_fd(current);
    if (next < 0) return -1;
    if (close_status != LF_APPROVED_OK) {
      (void)lf_close_fd(next);
      *status = LF_APPROVED_FAILED;
      return -1;
    }
    current = next;
    component_start = cursor + 1;
  }
  *status = LF_APPROVED_OK;
  return current;
}
