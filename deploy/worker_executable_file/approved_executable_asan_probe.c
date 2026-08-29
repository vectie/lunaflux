#define _DARWIN_C_SOURCE 1
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <linux/memfd.h>
#endif

#include "../../internal/process/process_approved_spawn.h"

static int snapshot_action = 0;
static int pread_failure = 0;
static int pin_failure = 0;
static int close_eintr = 0;
static int close_poison = 0;
static const char *snapshot_path = NULL;

static int probe_close(int fd) {
  if (close_eintr) {
    close_eintr = 0;
    errno = EINTR;
    return -1;
  }
  int status = close(fd);
  if (close_poison) {
    close_poison = 0;
    errno = EIO;
    return -1;
  }
  return status;
}

static ssize_t probe_pread(int fd, void *bytes, size_t length, off_t offset) {
  if (pread_failure) {
    pread_failure = 0;
    errno = EIO;
    return -1;
  }
  return pread(fd, bytes, length, offset);
}

static void probe_snapshot_hook(int stage, int fd) {
  if (stage == 1 && snapshot_action == 1) {
    assert(chmod(snapshot_path, 0700) == 0);
    int mutation = open(snapshot_path, O_WRONLY | O_TRUNC);
    assert(mutation >= 0 && close(mutation) == 0);
  }
  if (stage == 2 && snapshot_action == 2) {
    assert(chmod(snapshot_path, 0700) == 0);
    int mutation = open(snapshot_path, O_WRONLY | O_APPEND);
    assert(mutation >= 0 && write(mutation, "x", 1) == 1);
    assert(close(mutation) == 0);
  }
  if (stage == 2 && snapshot_action == 3) assert(close(fd) == 0);
}

#define LF_EXECUTABLE_CLOSE probe_close
#define LF_EXECUTABLE_PREAD probe_pread
#define LF_EXECUTABLE_SNAPSHOT_HOOK(stage, fd) probe_snapshot_hook((stage), (fd))
#define LF_EXECUTABLE_PIN_FAILURE() (pin_failure)
#include "approved_executable.c"

void *moonbit_make_external_object(
  void (*finalize)(void *self), uint32_t payload_size
) {
  (void)finalize;
  return calloc(1, payload_size);
}

static moonbit_bytes_t probe_array(int32_t length) {
  assert(length >= 0);
  struct moonbit_object *header = calloc(1, sizeof(*header) + (size_t)length);
  assert(header != NULL);
  /* Keep the standalone probe compatible with MoonBit SDKs before and after
   * the public Moonbit_make_dynamic_rc convenience macro. The object-header
   * bit layout and count shift are the stable ABI used by retain/release. */
  header->rc = (int32_t)(
    (1u << MOONBIT_RC_COUNT_SHIFT) |
    (uint32_t)moonbit_BLOCK_KIND_VAL_ARRAY
  );
  header->meta = (uint32_t)length;
  return (moonbit_bytes_t)(header + 1);
}

moonbit_bytes_t moonbit_make_bytes(int32_t size, int value) {
  moonbit_bytes_t result = probe_array(size);
  memset(result, value, (size_t)size);
  return result;
}

void moonbit_incref(void *object) {
  if (object != NULL) Moonbit_object_header(object)->rc += 1 << MOONBIT_RC_COUNT_SHIFT;
}

void moonbit_decref(void *object) {
  if (object == NULL) return;
  struct moonbit_object *header = Moonbit_object_header(object);
  header->rc -= 1 << MOONBIT_RC_COUNT_SHIFT;
  if (Moonbit_rc_count(header) == 0) free(header);
}

static moonbit_bytes_t probe_bytes(const char *value) {
  size_t length = strlen(value);
  assert(length <= INT32_MAX);
  moonbit_bytes_t result = probe_array((int32_t)length);
  memcpy(result, value, length);
  return result;
}

static int fd_count(void) {
  struct rlimit limit;
  assert(getrlimit(RLIMIT_NOFILE, &limit) == 0);
  rlim_t upper = limit.rlim_cur > 65536 ? 65536 : limit.rlim_cur;
  int count = 0;
  for (int fd = 0; (rlim_t)fd < upper; fd++) {
    errno = 0;
    if (fcntl(fd, F_GETFD) >= 0 || errno != EBADF) count++;
  }
  return count;
}

static void write_fixture(const char *path, const char *bytes) {
  (void)chmod(path, 0700);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0500);
  assert(fd >= 0);
  size_t length = strlen(bytes);
  assert(write(fd, bytes, length) == (ssize_t)length);
  assert(fchmod(fd, 0500) == 0);
  assert(close(fd) == 0);
}

#if defined(__linux__)
static void copy_fixture(const char *source, const char *destination) {
  int input = open(source, O_RDONLY | O_CLOEXEC);
  assert(input >= 0);
  struct stat info;
  assert(fstat(input, &info) == 0 && info.st_size > 0);
  /* A preceding immutable-snapshot case deliberately leaves this inode 0500.
   * Restore owner write authority before replacing the test namespace bytes. */
  assert(chmod(destination, 0700) == 0);
  int output = open(destination, O_WRONLY | O_CREAT | O_TRUNC, 0500);
  assert(output >= 0);
  char buffer[16384];
  off_t copied = 0;
  while (copied < info.st_size) {
    ssize_t count = read(input, buffer, sizeof(buffer));
    assert(count > 0);
    ssize_t cursor = 0;
    while (cursor < count) {
      ssize_t written = write(
        output, buffer + cursor, (size_t)(count - cursor)
      );
      assert(written > 0);
      cursor += written;
    }
    copied += count;
  }
  assert(fchmod(output, 0500) == 0);
  assert(close(output) == 0);
  assert(close(input) == 0);
}
#endif

static lf_approved_executable *open_owner(const char *path) {
  moonbit_bytes_t encoded = probe_bytes(path);
  int32_t status = -1;
  lf_approved_executable *owner = lunaflux_worker_executable_open(
    encoded, 1048576, &status
  );
  moonbit_decref(encoded);
  assert(owner != NULL && status == LF_EXECUTABLE_OK);
  return owner;
}

static void release_owner(lf_approved_executable *owner) {
  int32_t status = lunaflux_worker_executable_close(owner);
  assert(status == LF_EXECUTABLE_OK || status == LF_EXECUTABLE_FAILED);
  free(owner);
}

static void expect_snapshot_failure(
  const char *path, int action, int read_error, int pin_error,
  int close_error, int expected
) {
  write_fixture(path, "authenticated-worker");
  lf_approved_executable *owner = open_owner(path);
  moonbit_bytes_t failure = probe_bytes("failure");
  int32_t baseline_rc = Moonbit_object_header(failure)->rc;
  snapshot_action = action;
  snapshot_path = path;
  pread_failure = read_error;
  pin_failure = pin_error;
  close_poison = close_error;
  int32_t status = -1;
  moonbit_bytes_t result = lunaflux_worker_executable_snapshot_and_pin(
    owner, 1048576, failure, &status
  );
  snapshot_action = 0;
  snapshot_path = NULL;
  pin_failure = 0;
  close_poison = 0;
  assert(result == failure && status == expected);
  assert(Moonbit_object_header(failure)->rc ==
         baseline_rc + (1 << MOONBIT_RC_COUNT_SHIFT));
  moonbit_decref(result);
  assert(Moonbit_object_header(failure)->rc == baseline_rc);
  moonbit_decref(failure);
  release_owner(owner);
}

#if defined(__linux__)
typedef struct duplicate_stress {
  lf_approved_executable *owner;
  atomic_int failures;
} duplicate_stress;

static void *duplicate_worker(void *opaque) {
  duplicate_stress *stress = (duplicate_stress *)opaque;
  for (int iteration = 0; iteration < 2000; iteration++) {
    int32_t duplicate = -1;
    int32_t status = lf_approved_executable_duplicate(
      stress->owner, &duplicate
    );
    if (status == LF_APPROVED_CAPABILITY_OK) {
      if (duplicate < 5 || close(duplicate) != 0) {
        (void)atomic_fetch_add_explicit(
          &stress->failures, 1, memory_order_relaxed
        );
      }
    } else if (status != LF_APPROVED_CAPABILITY_CLOSED &&
               status != LF_APPROVED_CAPABILITY_BUSY) {
      (void)atomic_fetch_add_explicit(
        &stress->failures, 1, memory_order_relaxed
      );
    }
  }
  return NULL;
}

static void expect_pinned_spawn(lf_approved_executable *owner) {
  int32_t executable = -1;
  assert(lf_approved_executable_duplicate(owner, &executable) ==
         LF_APPROVED_CAPABILITY_OK);
  int model_source = open("/tmp", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  int kernel_source = open("/tmp", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  int channels[2] = {-1, -1};
  assert(model_source >= 0 && kernel_source >= 0);
  assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, channels) == 0);
  int child_channel = fcntl(channels[1], F_DUPFD_CLOEXEC, 6);
  int model = fcntl(model_source, F_DUPFD_CLOEXEC, 6);
  int kernel = fcntl(kernel_source, F_DUPFD_CLOEXEC, 6);
  assert(child_channel >= 6 && model >= 6 && kernel >= 6);
  assert(close(channels[1]) == 0);
  assert(close(model_source) == 0);
  assert(close(kernel_source) == 0);
  pid_t pid = -1;
  assert(lf_process_spawn_approved(
    executable, child_channel, model, kernel, &pid
  ) == 0);
  assert(close(executable) == 0);
  assert(close(child_channel) == 0);
  assert(close(model) == 0);
  assert(close(kernel) == 0);
  char response[5];
  assert(read(channels[0], response, sizeof(response)) == 5);
  assert(memcmp(response, "clean", sizeof(response)) == 0);
  assert(close(channels[0]) == 0);
  int child_status = 0;
  assert(waitpid(pid, &child_status, 0) == pid);
  assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);
}
#endif

int main(int argc, char **argv) {
  int before = fd_count();
  int32_t status = 99;
  lf_approved_executable *null_owner = lunaflux_worker_executable_open(
    NULL, 1024, &status
  );
  assert(status == LF_EXECUTABLE_INVALID);
  release_owner(null_owner);
  assert(lunaflux_worker_executable_snapshot_and_pin(
    NULL, -1, NULL, NULL
  ) == NULL);

#if defined(__APPLE__)
  char directory[] = "/private/tmp/lunaflux-approved-exec-XXXXXX";
#else
  char directory[] = "/tmp/lunaflux-approved-exec-XXXXXX";
#endif
  assert(mkdtemp(directory) != NULL);
  char path[512];
  char hardlink_path[512];
  char symlink_path[512];
  assert(snprintf(path, sizeof(path), "%s/worker", directory) > 0);
  assert(snprintf(hardlink_path, sizeof(hardlink_path), "%s/hardlink", directory) > 0);
  assert(snprintf(symlink_path, sizeof(symlink_path), "%s/symlink", directory) > 0);

  expect_snapshot_failure(path, 1, 0, 0, 0, LF_EXECUTABLE_TRUNCATED);
  expect_snapshot_failure(path, 0, 1, 0, 0, LF_EXECUTABLE_FAILED);
  expect_snapshot_failure(path, 2, 0, 0, 0, LF_EXECUTABLE_CHANGED);
  expect_snapshot_failure(path, 3, 0, 0, 0, LF_EXECUTABLE_FAILED);
#if defined(__linux__)
  assert(argc == 2);
  expect_snapshot_failure(path, 0, 0, 1, 0, LF_EXECUTABLE_FAILED);
  expect_snapshot_failure(path, 0, 0, 0, 1, LF_EXECUTABLE_FAILED);
#endif

  write_fixture(path, "authenticated-worker");
  lf_approved_executable *owner = open_owner(path);
  moonbit_bytes_t failure = probe_bytes("failure");
  moonbit_bytes_t snapshot = lunaflux_worker_executable_snapshot_and_pin(
    owner, 1048576, failure, &status
  );
  assert(status == LF_EXECUTABLE_OK);
  assert(Moonbit_array_length(snapshot) == 20);
  assert(memcmp(snapshot, "authenticated-worker", 20) == 0);
#if defined(__linux__)
  assert(link(path, hardlink_path) == 0);
  assert(chmod(hardlink_path, 0700) == 0);
  write_fixture(hardlink_path, "substituted-worker");
  int32_t duplicate = -1;
  assert(lf_approved_executable_duplicate(owner, &duplicate) ==
         LF_APPROVED_CAPABILITY_OK);
  char retained[32] = {0};
  assert(pread(duplicate, retained, 20, 0) == 20);
  assert(memcmp(retained, "authenticated-worker", 20) == 0);
  assert(fcntl(duplicate, F_GET_SEALS) ==
         (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL));
  assert(close(duplicate) == 0);
#endif
  moonbit_decref(snapshot);
  moonbit_decref(failure);

#if defined(__linux__)
  duplicate_stress stress = {.owner = owner, .failures = 0};
  pthread_t threads[4];
  for (int index = 0; index < 4; index++) {
    assert(pthread_create(&threads[index], NULL, duplicate_worker, &stress) == 0);
  }
  int close_status;
  do {
    close_status = lunaflux_worker_executable_close(owner);
  } while (close_status == LF_EXECUTABLE_BUSY);
  assert(close_status == LF_EXECUTABLE_OK);
  for (int index = 0; index < 4; index++) assert(pthread_join(threads[index], NULL) == 0);
  assert(atomic_load_explicit(&stress.failures, memory_order_relaxed) == 0);
  assert(lunaflux_worker_executable_close(owner) == LF_EXECUTABLE_OK);
  free(owner);

  copy_fixture(argv[1], path);
  owner = open_owner(path);
  failure = probe_bytes("failure");
  snapshot = lunaflux_worker_executable_snapshot_and_pin(
    owner, 67108864, failure, &status
  );
  assert(status == LF_EXECUTABLE_OK);
  write_fixture(path, "namespace-substitution");
  expect_pinned_spawn(owner);
  moonbit_decref(snapshot);
  moonbit_decref(failure);
  release_owner(owner);
#else
  (void)argc;
  (void)argv;
  int leased_fd = -1;
  assert(lf_exec_begin(owner, &leased_fd) == LF_EXECUTABLE_OK);
  assert(leased_fd >= 0);
  assert(lunaflux_worker_executable_close(owner) == LF_EXECUTABLE_BUSY);
  lf_exec_end(owner);
  assert(lunaflux_worker_executable_close(owner) == LF_EXECUTABLE_OK);
  free(owner);
#endif

  write_fixture(path, "authenticated-worker");
  owner = open_owner(path);
  close_poison = 1;
  assert(lunaflux_worker_executable_close(owner) == LF_EXECUTABLE_FAILED);
  assert(lunaflux_worker_executable_close(owner) == LF_EXECUTABLE_OK);
  free(owner);

  assert(symlink(path, symlink_path) == 0);
  moonbit_bytes_t symlink_bytes = probe_bytes(symlink_path);
  owner = lunaflux_worker_executable_open(symlink_bytes, 1024, &status);
  assert(status == LF_EXECUTABLE_UNSUPPORTED);
  moonbit_decref(symlink_bytes);
  release_owner(owner);

  assert(unlink(symlink_path) == 0);
#if defined(__linux__)
  assert(unlink(hardlink_path) == 0);
#endif
  assert(unlink(path) == 0);
  assert(rmdir(directory) == 0);
  assert(fd_count() == before);
  return 0;
}
