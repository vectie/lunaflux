#define _DARWIN_C_SOURCE 1
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_worker_executable_fixture_cat_path(void) {
#if defined(__linux__)
  static const char path[] = "/usr/bin/cat";
#else
  static const char path[] = "/bin/cat";
#endif
  moonbit_bytes_t result = moonbit_make_bytes((int32_t)(sizeof(path) - 1), 0);
  memcpy(result, path, sizeof(path) - 1);
  return result;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_worker_executable_fixture_read(
  moonbit_bytes_t path,
  int32_t *status
) {
  if (status == NULL) return moonbit_make_bytes(0, 0);
  *status = 1;
  if (path == NULL) return moonbit_make_bytes(0, 0);
  int32_t length = Moonbit_array_length(path);
  if (length < 4 || length > 4096 || path[0] != '/' ||
      memchr(path, '\0', (size_t)length) != NULL) {
    return moonbit_make_bytes(0, 0);
  }
  char path_copy[4097];
  memcpy(path_copy, path, (size_t)length);
  path_copy[length] = '\0';
  int fd = open(path_copy, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return moonbit_make_bytes(0, 0);
  struct stat info;
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
      info.st_size > 67108864 || info.st_size > INT32_MAX) {
    close(fd);
    return moonbit_make_bytes(0, 0);
  }
  int32_t byte_count = (int32_t)info.st_size;
  moonbit_bytes_t result = moonbit_make_bytes(byte_count, 0);
  int32_t cursor = 0;
  while (cursor < byte_count) {
    ssize_t count = pread(
      fd, result + cursor, (size_t)(byte_count - cursor), (off_t)cursor
    );
    if (count > 0) cursor += (int32_t)count;
    else if (count < 0 && errno == EINTR) continue;
    else {
      close(fd);
      *status = 2;
      return result;
    }
  }
  if (close(fd) != 0) {
    *status = 2;
    return result;
  }
  *status = 0;
  return result;
}
