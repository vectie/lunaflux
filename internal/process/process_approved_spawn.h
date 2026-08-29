#ifndef LUNAFLUX_PROCESS_APPROVED_SPAWN_H
#define LUNAFLUX_PROCESS_APPROVED_SPAWN_H

#include <sys/types.h>

/* Linux-only descriptor execution. All descriptors are borrowed for the
 * duration of this call; the caller retains and closes them in the parent. */
int lf_process_spawn_approved(
  int executable_fd,
  int channel_fd,
  int model_root_fd,
  int kernel_root_fd,
  pid_t *pid_out
);

#endif
