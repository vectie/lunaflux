#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include "process_approved_spawn.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

int lf_process_spawn_approved(
  int executable_fd,
  int channel_fd,
  int model_root_fd,
  int kernel_root_fd,
  pid_t *pid_out
) {
  if (executable_fd < 5 || channel_fd < 5 || model_root_fd < 5 ||
      kernel_root_fd < 5 || executable_fd == channel_fd ||
      executable_fd == model_root_fd || executable_fd == kernel_root_fd ||
      channel_fd == model_root_fd || channel_fd == kernel_root_fd ||
      model_root_fd == kernel_root_fd || pid_out == NULL) {
    return EINVAL;
  }
#if defined(__linux__)
  sigset_t all_signals;
  sigset_t parent_mask;
  if (sigfillset(&all_signals) != 0) return errno;
  int block_status = pthread_sigmask(SIG_BLOCK, &all_signals, &parent_mask);
  if (block_status != 0) return block_status;
  pid_t pid = fork();
  int fork_error = errno;
  if (pid != 0) {
    int restore_status = pthread_sigmask(SIG_SETMASK, &parent_mask, NULL);
    if (pid < 0) return fork_error;
    if (restore_status != 0) {
      (void)kill(pid, SIGKILL);
      while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
      }
      return restore_status;
    }
  }
  if (pid == 0) {
    sigset_t empty_mask;
    struct sigaction default_action;
    default_action.sa_handler = SIG_DFL;
    default_action.sa_flags = 0;
    if (sigemptyset(&empty_mask) != 0 ||
        sigemptyset(&default_action.sa_mask) != 0) {
      _exit(126);
    }
    for (int signal_number = 1; signal_number < NSIG; signal_number++) {
      if (signal_number == SIGKILL || signal_number == SIGSTOP) continue;
      if (sigaction(signal_number, &default_action, NULL) != 0 &&
          errno != EINVAL) {
        _exit(126);
      }
    }
    if (close(2) != 0 && errno != EBADF) _exit(126);
    if (dup2(channel_fd, 0) < 0 || dup2(channel_fd, 1) < 0 ||
        dup2(model_root_fd, 3) < 0 ||
        dup2(kernel_root_fd, 4) < 0 || dup2(executable_fd, 5) < 0 ||
        fcntl(5, F_SETFD, FD_CLOEXEC) < 0) {
      _exit(126);
    }
    if (syscall(SYS_close_range, 6u, UINT_MAX, 0u) != 0) {
      _exit(126);
    }
    char *const argv[] = {(char *)"lunaflux-worker", NULL};
    char *const sanitized_environment[] = {NULL};
    if (sigprocmask(SIG_SETMASK, &empty_mask, NULL) != 0) _exit(126);
    fexecve(5, argv, sanitized_environment);
    _exit(127);
  }
  *pid_out = pid;
  return 0;
#else
  return ENOTSUP;
#endif
}
