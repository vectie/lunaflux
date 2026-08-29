#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

extern char **environ;

int main(void) {
  sigset_t mask;
  struct sigaction action;
  struct stat info;
  if (environ == NULL || environ[0] != NULL ||
      sigprocmask(SIG_SETMASK, NULL, &mask) != 0 ||
      sigismember(&mask, SIGTERM) != 0 || sigismember(&mask, SIGUSR1) != 0 ||
      sigaction(SIGTERM, NULL, &action) != 0 || action.sa_handler != SIG_DFL ||
      sigaction(SIGUSR1, NULL, &action) != 0 || action.sa_handler != SIG_DFL ||
      fstat(3, &info) != 0 || !S_ISDIR(info.st_mode) ||
      fstat(4, &info) != 0 || !S_ISDIR(info.st_mode)) {
    return 2;
  }
  errno = 0;
  if (fcntl(5, F_GETFD) >= 0 || errno != EBADF) return 3;
  errno = 0;
  if (fcntl(2, F_GETFD) >= 0 || errno != EBADF) return 4;
  errno = 0;
  if (write(2, "stderr-must-be-closed", 21) >= 0 || errno != EBADF) return 5;
  errno = 0;
  if (fcntl(900, F_GETFD) >= 0 || errno != EBADF) return 6;
  return write(1, "clean", 5) == 5 ? 0 : 7;
}
