#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include "process_approved_spawn.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
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
static void inherited_handler(int signal_number) { (void)signal_number; }

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

static int spawn_once(const char *executable, int sentinel) {
  int executable_fd = open(executable, O_RDONLY | O_CLOEXEC);
  int model_fd = open("/tmp", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  int kernel_fd = open("/tmp", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  int channels[2] = {-1, -1};
  assert(executable_fd >= 0 && model_fd >= 0 && kernel_fd >= 0);
  assert(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, channels) == 0);
  int moved_exec = fcntl(executable_fd, F_DUPFD_CLOEXEC, 6);
  int moved_channel = fcntl(channels[1], F_DUPFD_CLOEXEC, 6);
  int moved_model = fcntl(model_fd, F_DUPFD_CLOEXEC, 6);
  int moved_kernel = fcntl(kernel_fd, F_DUPFD_CLOEXEC, 6);
  assert(moved_exec >= 6 && moved_channel >= 6 && moved_model >= 6 &&
         moved_kernel >= 6);
  close(executable_fd);
  close(channels[1]);
  close(model_fd);
  close(kernel_fd);

  int sentinel_fd = -1;
  struct rlimit original_limit;
  assert(getrlimit(RLIMIT_NOFILE, &original_limit) == 0);
  if (sentinel && original_limit.rlim_max > 900) {
    int source = open("/dev/null", O_RDONLY);
    assert(source >= 0);
    (void)close(900);
    sentinel_fd = fcntl(source, F_DUPFD, 900);
    assert(sentinel_fd == 900);
    close(source);
    struct rlimit lowered = original_limit;
    lowered.rlim_cur = 256;
    assert(setrlimit(RLIMIT_NOFILE, &lowered) == 0);
  }

  struct sigaction ignored = {.sa_handler = SIG_IGN, .sa_flags = 0};
  struct sigaction caught = {.sa_handler = inherited_handler, .sa_flags = 0};
  struct sigaction old_term;
  struct sigaction old_usr1;
  assert(sigemptyset(&ignored.sa_mask) == 0);
  assert(sigemptyset(&caught.sa_mask) == 0);
  assert(sigaction(SIGTERM, &ignored, &old_term) == 0);
  assert(sigaction(SIGUSR1, &caught, &old_usr1) == 0);
  sigset_t blocked;
  sigset_t old_mask;
  assert(sigemptyset(&blocked) == 0);
  assert(sigaddset(&blocked, SIGTERM) == 0);
  assert(sigaddset(&blocked, SIGUSR1) == 0);
  assert(sigprocmask(SIG_BLOCK, &blocked, &old_mask) == 0);

  pid_t pid = -1;
  int spawned = lf_process_spawn_approved(
    moved_exec, moved_channel, moved_model, moved_kernel, &pid
  );
  assert(spawned == 0 && pid > 0);
  struct sigaction observed;
  sigset_t observed_mask;
  assert(sigaction(SIGTERM, NULL, &observed) == 0 &&
         observed.sa_handler == SIG_IGN);
  assert(sigprocmask(SIG_SETMASK, NULL, &observed_mask) == 0 &&
         sigismember(&observed_mask, SIGTERM) == 1);
  assert(sigprocmask(SIG_SETMASK, &old_mask, NULL) == 0);
  assert(sigaction(SIGTERM, &old_term, NULL) == 0);
  assert(sigaction(SIGUSR1, &old_usr1, NULL) == 0);
  close(moved_exec);
  close(moved_channel);
  close(moved_model);
  close(moved_kernel);
  char result[5];
  ssize_t count = read(channels[0], result, sizeof(result));
  close(channels[0]);
  int status = 0;
  assert(waitpid(pid, &status, 0) == pid);
  if (sentinel_fd >= 0) {
    assert(setrlimit(RLIMIT_NOFILE, &original_limit) == 0);
    assert(close(sentinel_fd) == 0);
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0 && count == 5 &&
    memcmp(result, "clean", 5) == 0;
}
#endif

int main(int argc, char **argv) {
#if defined(__linux__)
  assert(argc == 2);
  int before = fd_count();
  assert(spawn_once(argv[1], 1));
  for (int iteration = 0; iteration < 24; iteration++) {
    assert(spawn_once(argv[1], 0));
  }
  assert(fd_count() == before);
#else
  (void)argc;
  (void)argv;
#endif
  return 0;
}
