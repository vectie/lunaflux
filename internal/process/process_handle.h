#ifndef LUNAFLUX_PROCESS_HANDLE_H
#define LUNAFLUX_PROCESS_HANDLE_H

#include <sys/types.h>

typedef struct lf_process {
  pid_t pid;
  int fd;
  int reaped;
  int closed;
  int exit_kind;
  int exit_code;
} lf_process;

#endif
