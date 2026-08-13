#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int probe_kill_failure = 0;
static int probe_wait_failure = 0;
static int probe_close_evidence = 0;
static int probe_fcntl_failure = 0;

static int probe_kill(pid_t pid, int signal_number) {
  if (probe_kill_failure) {
    errno = EPERM;
    return -1;
  }
  return kill(pid, signal_number);
}

static pid_t probe_waitpid(pid_t pid, int *status, int options) {
  if (probe_wait_failure) {
    errno = EIO;
    return -1;
  }
  return waitpid(pid, status, options);
}

static int probe_close(int fd) {
  int result = close(fd);
  if (probe_close_evidence) {
    errno = EIO;
    return -1;
  }
  return result;
}

static int probe_fcntl(int fd, int command, int value) {
  if (probe_fcntl_failure) {
    errno = EIO;
    return -1;
  }
  return fcntl(fd, command, value);
}

#define LF_PROCESS_KILL probe_kill
#define LF_PROCESS_WAITPID probe_waitpid
#define LF_PROCESS_CLOSE probe_close
#define LF_PROCESS_FCNTL probe_fcntl

#include "process.c"

void *moonbit_make_external_object(
  void (*finalize)(void *self), uint32_t payload_size
) {
  (void)finalize;
  return calloc(1, payload_size);
}

void moonbit_incref(void *object) { (void)object; }

static moonbit_bytes_t probe_bytes(const char *value, int terminate) {
  size_t source = strlen(value);
  size_t length = source + (terminate ? 1u : 0u);
  struct moonbit_object *header = calloc(1, sizeof(*header) + length);
  assert(header != NULL);
  header->meta = (uint32_t)length;
  moonbit_bytes_t bytes = (moonbit_bytes_t)(header + 1);
  memcpy(bytes, value, source);
  return bytes;
}

static void probe_free_bytes(moonbit_bytes_t bytes) {
  free(Moonbit_object_header(bytes));
}

static moonbit_bytes_t probe_raw(const uint8_t *value, size_t length) {
  struct moonbit_object *header = calloc(1, sizeof(*header) + length);
  assert(header != NULL);
  header->meta = (uint32_t)length;
  moonbit_bytes_t bytes = (moonbit_bytes_t)(header + 1);
  if (length > 0) memcpy(bytes, value, length);
  return bytes;
}

int main(void) {
  lf_process *null_path = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared(null_path, NULL) != LF_PROCESS_OK);
  assert(lunaflux_process_close(null_path) == LF_PROCESS_OK);

  const uint8_t empty_value[] = {0};
  moonbit_bytes_t empty = probe_raw(empty_value, 0);
  moonbit_bytes_t one = probe_raw(empty_value, 1);
  lf_process *empty_child = lunaflux_process_prepare_child();
  lf_process *one_child = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared(empty_child, empty) != LF_PROCESS_OK);
  assert(lunaflux_process_spawn_prepared(one_child, one) != LF_PROCESS_OK);
  assert(lunaflux_process_close(empty_child) == LF_PROCESS_OK);
  assert(lunaflux_process_close(one_child) == LF_PROCESS_OK);
  probe_free_bytes(empty);
  probe_free_bytes(one);

  const uint8_t early_nul_value[] = {'/', 'b', 0, 'x', 0};
  const uint8_t no_final_nul_value[] = {'/', 'b', 'i', 'n'};
  moonbit_bytes_t early_nul = probe_raw(early_nul_value, sizeof(early_nul_value));
  moonbit_bytes_t no_final_nul = probe_raw(
    no_final_nul_value, sizeof(no_final_nul_value)
  );
  lf_process *early_child = lunaflux_process_prepare_child();
  lf_process *final_child = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared(early_child, early_nul) != LF_PROCESS_OK);
  assert(lunaflux_process_spawn_prepared(final_child, no_final_nul) != LF_PROCESS_OK);
  assert(lunaflux_process_close(early_child) == LF_PROCESS_OK);
  assert(lunaflux_process_close(final_child) == LF_PROCESS_OK);
  probe_free_bytes(early_nul);
  probe_free_bytes(no_final_nul);

  moonbit_bytes_t unterminated = probe_bytes("/bin/true", 0);
  lf_process *bad_path = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared(bad_path, unterminated) != LF_PROCESS_OK);
  assert(lunaflux_process_close(bad_path) == LF_PROCESS_OK);
  probe_free_bytes(unterminated);

#if defined(__APPLE__)
  moonbit_bytes_t valid = probe_bytes("/usr/bin/true", 1);
#else
  moonbit_bytes_t valid = probe_bytes("/bin/true", 1);
#endif
  lf_process *null_roots = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared_with_approved_roots(
    null_roots, valid, NULL
  ) != LF_PROCESS_OK);
  assert(lunaflux_process_close(null_roots) == LF_PROCESS_OK);

  lf_process *child = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared(child, valid) == LF_PROCESS_OK);
  assert(lunaflux_process_spawn_prepared(child, valid) != LF_PROCESS_OK);
  assert(lunaflux_process_close(child) == LF_PROCESS_OK);
  assert(lunaflux_process_close(child) == LF_PROCESS_OK);
  probe_free_bytes(valid);

#if defined(__APPLE__)
  valid = probe_bytes("/usr/bin/true", 1);
#else
  valid = probe_bytes("/bin/true", 1);
#endif
  lf_process *fcntl_failed = lunaflux_process_prepare_child();
  probe_fcntl_failure = 1;
  assert(lunaflux_process_spawn_prepared(fcntl_failed, valid) == LF_PROCESS_FAILED);
  probe_fcntl_failure = 0;
  assert(lunaflux_process_is_closed(fcntl_failed));
  assert(fcntl_failed->pid == -1);
  assert(lunaflux_process_close(fcntl_failed) == LF_PROCESS_OK);
  probe_free_bytes(valid);

  lf_process *kill_failed = lunaflux_process_prepare_child();
  kill_failed->pid = 999999;
  kill_failed->reaped = 0;
  kill_failed->closed = 0;
  probe_kill_failure = 1;
  assert(lunaflux_process_close(kill_failed) == LF_PROCESS_FAILED);
  assert(!lunaflux_process_is_closed(kill_failed));
  probe_kill_failure = 0;
  assert(lunaflux_process_close(kill_failed) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(kill_failed));

  lf_process *wait_failed = lunaflux_process_prepare_child();
  wait_failed->pid = 999998;
  wait_failed->reaped = 0;
  wait_failed->closed = 0;
  probe_wait_failure = 1;
  assert(lunaflux_process_close(wait_failed) == LF_PROCESS_FAILED);
  assert(!lunaflux_process_is_closed(wait_failed));
  probe_wait_failure = 0;
  assert(lunaflux_process_close(wait_failed) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(wait_failed));

  int pipe_fds[2];
  assert(pipe(pipe_fds) == 0);
  lf_process *close_evidence = lunaflux_process_prepare_child();
  close_evidence->fd = pipe_fds[0];
  close_evidence->closed = 0;
  probe_close_evidence = 1;
  assert(lunaflux_process_close(close_evidence) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(close_evidence));
  probe_close_evidence = 0;
  assert(lunaflux_process_close(close_evidence) == LF_PROCESS_OK);
  assert(close(pipe_fds[1]) == 0);
  return 0;
}
