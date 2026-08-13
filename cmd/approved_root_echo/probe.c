#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/stat.h>

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_root_probe_fd_closed(int32_t fd) {
  errno = 0;
  return fcntl(fd, F_GETFD, 0) < 0 && errno == EBADF;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_root_probe_environment_empty(void) {
  return getenv("LUNAFLUX_INHERITANCE_SENTINEL") == NULL;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_root_probe_directory_fds_cloexec(void) {
  struct rlimit limit;
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return 0;
  rlim_t upper = limit.rlim_cur > 65536 ? 65536 : limit.rlim_cur;
  int directories = 0;
  for (int fd = 5; (rlim_t)fd < upper; fd++) {
    int flags = fcntl(fd, F_GETFD, 0);
    if (flags < 0) continue;
    struct stat info;
    if (fstat(fd, &info) != 0 || !S_ISDIR(info.st_mode)) continue;
    directories++;
    if ((flags & FD_CLOEXEC) == 0) return 0;
  }
  return directories >= 2;
}
