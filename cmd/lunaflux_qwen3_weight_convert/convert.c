#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_all(int fd, const uint8_t *bytes, size_t length) {
  size_t offset = 0;
  while (offset < length) {
    ssize_t written = write(fd, bytes + offset, length - offset);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return -1;
    offset += (size_t)written;
  }
  return 0;
}

static int close_fd(int fd) {
#if defined(__APPLE__)
  int status;
  do status = close(fd); while (status != 0 && errno == EINTR);
  return status;
#else
  return close(fd);
#endif
}

static uint64_t read_u64_le(const uint8_t *bytes) {
  uint64_t value = 0;
  for (int index = 0; index < 8; index++) {
    value |= ((uint64_t)bytes[index]) << (index * 8);
  }
  return value;
}

static char *copy_path(moonbit_bytes_t bytes) {
  int32_t length = Moonbit_array_length(bytes);
  if (length <= 0 || memchr(bytes, '\0', (size_t)length) != NULL) return NULL;
  char *path = (char *)malloc((size_t)length + 1u);
  if (path == NULL) return NULL;
  memcpy(path, bytes, (size_t)length);
  path[length] = '\0';
  return path;
}

static int same_stamp(const struct stat *left, const struct stat *right) {
#if defined(__APPLE__)
  return left->st_dev == right->st_dev && left->st_ino == right->st_ino &&
         left->st_size == right->st_size &&
         left->st_mtimespec.tv_sec == right->st_mtimespec.tv_sec &&
         left->st_mtimespec.tv_nsec == right->st_mtimespec.tv_nsec &&
         left->st_ctimespec.tv_sec == right->st_ctimespec.tv_sec &&
         left->st_ctimespec.tv_nsec == right->st_ctimespec.tv_nsec;
#else
  return left->st_dev == right->st_dev && left->st_ino == right->st_ino &&
         left->st_size == right->st_size &&
         left->st_mtim.tv_sec == right->st_mtim.tv_sec &&
         left->st_mtim.tv_nsec == right->st_mtim.tv_nsec &&
         left->st_ctim.tv_sec == right->st_ctim.tv_sec &&
         left->st_ctim.tv_nsec == right->st_ctim.tv_nsec;
#endif
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_convert_qwen3_numeric_file(
    moonbit_bytes_t source_path, moonbit_bytes_t output_path,
    moonbit_bytes_t header, moonbit_bytes_t regions, int32_t chunk_bytes) {
  if (chunk_bytes <= 0 || Moonbit_array_length(header) == 0 ||
      Moonbit_array_length(header) % 8 != 0 ||
      Moonbit_array_length(regions) == 0 ||
      Moonbit_array_length(regions) % 16 != 0) return 1;
  char *source_name = copy_path(source_path);
  char *output_name = copy_path(output_path);
  if (source_name == NULL || output_name == NULL) {
    free(source_name);
    free(output_name);
    return 1;
  }
  int source = open(source_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (source < 0) {
    free(source_name);
    free(output_name);
    return 2;
  }
  struct stat before;
  if (fstat(source, &before) != 0 || !S_ISREG(before.st_mode)) {
    close_fd(source);
    free(source_name);
    free(output_name);
    return 2;
  }
  uint8_t prefix[8];
  if (pread(source, prefix, 8, 0) != 8) {
    close_fd(source);
    free(source_name);
    free(output_name);
    return 2;
  }
  uint64_t encoded_source_header = read_u64_le(prefix);
  if (encoded_source_header > (uint64_t)INT64_MAX - 8u ||
      encoded_source_header + 8u > (uint64_t)before.st_size) {
    close_fd(source);
    free(source_name);
    free(output_name);
    return 2;
  }
  uint64_t source_payload = encoded_source_header + 8u;
  int output = open(output_name,
                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                    S_IRUSR);
  if (output < 0) {
    close_fd(source);
    free(source_name);
    free(output_name);
    return 3;
  }
  int status = 0;
  uint8_t output_prefix[8];
  uint64_t header_length = (uint64_t)Moonbit_array_length(header);
  for (int index = 0; index < 8; index++) {
    output_prefix[index] = (uint8_t)(header_length >> (index * 8));
  }
  if (write_all(output, output_prefix, 8) != 0 ||
      write_all(output, header, (size_t)header_length) != 0) status = 4;
  uint8_t *scratch = NULL;
  if (status == 0) {
    scratch = (uint8_t *)malloc((size_t)chunk_bytes);
    if (scratch == NULL) status = 4;
  }
  int32_t region_bytes = Moonbit_array_length(regions);
  for (int32_t cursor = 0; status == 0 && cursor < region_bytes; cursor += 16) {
    uint64_t relative = read_u64_le(regions + cursor);
    uint64_t length = read_u64_le(regions + cursor + 8);
    uint64_t available = (uint64_t)before.st_size - source_payload;
    if (relative > available || length > available - relative) {
      status = 4;
      break;
    }
    uint64_t copied = 0;
    while (copied < length) {
      size_t count = (size_t)((length - copied) < (uint64_t)chunk_bytes
                                  ? (length - copied)
                                  : (uint64_t)chunk_bytes);
      ssize_t received = pread(source, scratch, count,
                               (off_t)(source_payload + relative + copied));
      if (received < 0 && errno == EINTR) continue;
      if (received != (ssize_t)count || write_all(output, scratch, count) != 0) {
        status = 4;
        break;
      }
      copied += count;
    }
  }
  free(scratch);
  struct stat after;
  if (status == 0 &&
      (fstat(source, &after) != 0 || !same_stamp(&before, &after))) status = 5;
  if (status == 0 && fsync(output) != 0) status = 4;
  if (close_fd(output) != 0 && status == 0) status = 4;
  if (close_fd(source) != 0 && status == 0) status = 4;
  if (status != 0) unlink(output_name);
  free(source_name);
  free(output_name);
  return status;
}
