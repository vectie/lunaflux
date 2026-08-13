#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static int probe_snapshot_action = 0;
static const char *probe_snapshot_path = NULL;
static void probe_snapshot_hook(int stage, int fd);
#define LF_APPROVED_FS_SNAPSHOT_HOOK(stage, fd) \
  probe_snapshot_hook((stage), (fd))

/* Compile the exact production translation unit under ASan without linking
 * the MoonBit allocator. The tiny functions below model only the two runtime
 * allocation layouts used by this ABI probe. */
#include "approved_fs.c"

void *moonbit_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
) {
  (void)finalize;
  return calloc(1, payload_size);
}

void moonbit_incref(void *object) {
  (void)object;
}

static void *probe_array(size_t element_size, int32_t length) {
  size_t bytes = sizeof(struct moonbit_object) + element_size * (size_t)length;
  struct moonbit_object *header = (struct moonbit_object *)calloc(1, bytes);
  assert(header != NULL);
  header->meta = (uint32_t)length;
  return header + 1;
}

static void probe_array_free(void *value) {
  free(Moonbit_object_header(value));
}

moonbit_bytes_t moonbit_make_bytes(int32_t size, int value) {
  assert(size >= 0);
  moonbit_bytes_t result = (moonbit_bytes_t)probe_array(1, size);
  memset(result, value, (size_t)size);
  return result;
}

static moonbit_bytes_t probe_bytes(const char *value) {
  size_t length = strlen(value);
  assert(length <= INT32_MAX);
  moonbit_bytes_t result = (moonbit_bytes_t)probe_array(1, (int32_t)length);
  memcpy(result, value, length);
  return result;
}

static int32_t probe_fd_count(void) {
  struct rlimit limit;
  assert(getrlimit(RLIMIT_NOFILE, &limit) == 0);
  rlim_t upper = limit.rlim_cur > 65536 ? 65536 : limit.rlim_cur;
  int32_t count = 0;
  for (int fd = 0; (rlim_t)fd < upper; fd++) {
    errno = 0;
    if (fcntl(fd, F_GETFD) >= 0 || errno != EBADF) count++;
  }
  return count;
}

static void probe_handle_free(lf_approved_handle *handle) {
  assert(lunaflux_approved_fs_close(handle) == LF_APPROVED_OK);
  free(handle);
}

static void probe_write_exact_file(const char *path, const char *bytes) {
  int file = open(path, O_TRUNC | O_WRONLY | O_CLOEXEC);
  assert(file >= 0);
  assert(write(file, bytes, 3) == 3);
  assert(close(file) == 0);
}

static void probe_snapshot_hook(int stage, int fd) {
  (void)fd;
  if (stage != 1 || probe_snapshot_action == 0) return;
  assert(probe_snapshot_path != NULL);
  int mutator = open(probe_snapshot_path, O_WRONLY | O_CLOEXEC);
  assert(mutator >= 0);
  if (probe_snapshot_action == 1) {
    assert(ftruncate(mutator, 2) == 0);
  } else if (probe_snapshot_action == 2) {
    assert(pwrite(mutator, "d", 1, 3) == 1);
  } else {
    assert(probe_snapshot_action == 3);
    assert(pwrite(mutator, "z", 1, 0) == 1);
    struct stat info;
    assert(fstat(mutator, &info) == 0);
    struct timespec times[2];
#if defined(__APPLE__)
    times[0] = info.st_atimespec;
    times[1] = info.st_mtimespec;
#else
    times[0] = info.st_atim;
    times[1] = info.st_mtim;
#endif
    times[1].tv_sec += 1;
    assert(futimens(mutator, times) == 0);
  }
  assert(close(mutator) == 0);
}

int main(void) {
  char root_path[] = "/tmp/lunaflux-approved-fs-asan.XXXXXX";
  assert(mkdtemp(root_path) != NULL);
  char canonical_root[PATH_MAX];
  assert(realpath(root_path, canonical_root) != NULL);
  char file_path[sizeof(root_path) + 16];
  int written = snprintf(file_path, sizeof(file_path), "%s/file.bin", root_path);
  assert(written > 0 && (size_t)written < sizeof(file_path));
  int created = open(file_path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
  assert(created >= 0);
  assert(write(created, "abc", 3) == 3);
  assert(close(created) == 0);

  moonbit_bytes_t root_bytes = probe_bytes(canonical_root);
  moonbit_bytes_t file_bytes = probe_bytes("file.bin");
  moonbit_bytes_t escape_bytes = probe_bytes("../file.bin");
  moonbit_bytes_t snapshot_failure = probe_bytes("");
  probe_snapshot_path = file_path;
  int32_t before = probe_fd_count();

  for (int iteration = 0; iteration < 1024; iteration++) {
    int32_t status = -1;
    lf_approved_handle *root = lunaflux_approved_fs_open_root(root_bytes, &status);
    assert(status == LF_APPROVED_OK);
    lf_approved_handle *escape =
      lunaflux_approved_fs_open_file(root, escape_bytes, &status);
    assert(status == LF_APPROVED_INVALID);
    probe_handle_free(escape);
    lf_approved_handle *file =
      lunaflux_approved_fs_open_file(root, file_bytes, &status);
    assert(status == LF_APPROVED_OK);

    uint8_t *destination = (uint8_t *)probe_array(1, 5);
    memset(destination, '!', 5);
    assert(lunaflux_approved_fs_read_exact_at(file, destination, 1, 3, 0) == 0);
    assert(destination[0] == '!' && destination[1] == 'a');
    assert(destination[2] == 'b' && destination[3] == 'c');
    assert(destination[4] == '!');
    assert(lunaflux_approved_fs_read_exact_at(file, destination, 4, 2, 0) == 1);

    int64_t *stamp = (int64_t *)probe_array(sizeof(int64_t), 5);
    assert(lunaflux_approved_fs_stamp(file, stamp) == 0);
    assert(stamp[0] == 3);

    int32_t snapshot_status = -1;
    moonbit_bytes_t snapshot = lunaflux_approved_fs_read_immutable_snapshot(
      file, 3, snapshot_failure, &snapshot_status
    );
    assert(snapshot_status == LF_APPROVED_OK);
    assert(Moonbit_array_length(snapshot) == 3);
    assert(memcmp(snapshot, "abc", 3) == 0);
    probe_array_free(snapshot);

    snapshot = lunaflux_approved_fs_read_immutable_snapshot(
      file, 2, snapshot_failure, &snapshot_status
    );
    assert(snapshot_status == LF_APPROVED_TOO_LARGE);
    assert(snapshot == snapshot_failure);

    probe_snapshot_action = 1;
    snapshot = lunaflux_approved_fs_read_immutable_snapshot(
      file, 3, snapshot_failure, &snapshot_status
    );
    probe_snapshot_action = 0;
    assert(snapshot_status == LF_APPROVED_TRUNCATED);
    probe_array_free(snapshot);
    probe_write_exact_file(file_path, "abc");

    probe_snapshot_action = 2;
    snapshot = lunaflux_approved_fs_read_immutable_snapshot(
      file, 4, snapshot_failure, &snapshot_status
    );
    probe_snapshot_action = 0;
    assert(snapshot_status == LF_APPROVED_CHANGED);
    probe_array_free(snapshot);
    probe_write_exact_file(file_path, "abc");

    probe_snapshot_action = 3;
    snapshot = lunaflux_approved_fs_read_immutable_snapshot(
      file, 3, snapshot_failure, &snapshot_status
    );
    probe_snapshot_action = 0;
    assert(snapshot_status == LF_APPROVED_CHANGED);
    probe_array_free(snapshot);
    probe_write_exact_file(file_path, "abc");

    assert(lunaflux_approved_fs_test_close_while_active(file, 2) == 7);
    assert(lunaflux_approved_fs_is_closed(file) == 0);

    probe_array_free(destination);
    probe_array_free(stamp);
    probe_handle_free(file);
    probe_handle_free(root);
  }

  assert(probe_fd_count() == before);
  probe_array_free(root_bytes);
  probe_array_free(file_bytes);
  probe_array_free(escape_bytes);
  probe_array_free(snapshot_failure);
  assert(unlink(file_path) == 0);
  assert(rmdir(root_path) == 0);
  return 0;
}
