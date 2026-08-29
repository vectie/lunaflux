#define _DARWIN_C_SOURCE 1
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../../internal/process/process_handle.h"

extern char **environ;

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_test_spawn_fixture(
  lf_process *process,
  moonbit_bytes_t path
) {
  if (process == NULL || path == NULL || !process->closed || process->fd >= 0) {
    return 1;
  }
  int32_t length = Moonbit_array_length(path);
  if (length < 4 || length > 4096 || path[0] != '/' ||
      memchr(path, '\0', (size_t)length) != NULL) return 1;
  char path_copy[4097];
  memcpy(path_copy, path, (size_t)length);
  path_copy[length] = '\0';
  int raw_sockets[2] = {-1, -1};
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, raw_sockets) != 0) return 1;
  for (int index = 0; index < 2; index++) {
    sockets[index] = fcntl(raw_sockets[index], F_DUPFD_CLOEXEC, 5);
    if (sockets[index] < 0) {
      if (sockets[0] >= 0) (void)close(sockets[0]);
      (void)close(raw_sockets[0]);
      (void)close(raw_sockets[1]);
      return 1;
    }
  }
  (void)close(raw_sockets[0]);
  (void)close(raw_sockets[1]);
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int setup = posix_spawn_file_actions_init(&actions);
  if (setup == 0) setup = posix_spawnattr_init(&attributes);
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 0);
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 1);
  if (setup == 0) setup = posix_spawn_file_actions_addclose(&actions, 2);
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
  if (setup == 0) setup = posix_spawn_file_actions_addclose(&actions, 3);
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 4);
  if (setup == 0) setup = posix_spawn_file_actions_addclose(&actions, 4);
  if (setup == 0) setup = posix_spawn_file_actions_addclose(&actions, sockets[0]);
  if (setup == 0) setup = posix_spawn_file_actions_addclose(&actions, sockets[1]);
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  if (setup == 0) {
    setup = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
  }
#endif
  pid_t pid = -1;
  char *const argv[] = {path_copy, NULL};
  if (setup == 0) {
    setup = posix_spawn(&pid, path_copy, &actions, &attributes, argv, environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  (void)close(sockets[1]);
  if (setup != 0) {
    (void)close(sockets[0]);
    return 1;
  }
  int flags = fcntl(sockets[0], F_GETFL, 0);
  if (flags < 0 || fcntl(sockets[0], F_SETFL, flags | O_NONBLOCK) != 0) {
    (void)close(sockets[0]);
    (void)kill(pid, SIGKILL);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
    }
    return 1;
  }
  process->pid = pid;
  process->fd = sockets[0];
  process->reaped = 0;
  process->closed = 0;
  return 0;
}
