#define _GNU_SOURCE 1
#include "process_approved_spawn.h"
#include <assert.h>
#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* Tiny offline GPU test of the EXACT approved worker spawn implementation. */
int main(int argc, char **argv) {
  if (argc == 1) {
    struct stat a, b;
    assert(fstat(3, &a) == 0 && S_ISDIR(a.st_mode));
    assert(fstat(4, &b) == 0 && S_ISDIR(b.st_mode));
    assert(getenv("LUNAFLUX_UNRELATED_SENTINEL") == NULL);
    assert(getenv("NSYS_PROFILING_SESSION_ID") != NULL);
    CUcontext ctx;
    CUdevice device;
    CUdeviceptr p;
    assert(cuInit(0) == CUDA_SUCCESS);
    assert(cuDeviceGet(&device, 0) == CUDA_SUCCESS);
    assert(cuCtxCreate(&ctx, NULL, 0, device) == CUDA_SUCCESS);
    assert(cuMemAlloc(&p, 4096) == CUDA_SUCCESS);
    assert(cuMemsetD32(p, 17, 1024) == CUDA_SUCCESS);
    assert(cuCtxSynchronize() == CUDA_SUCCESS);
    assert(cuMemFree(p) == CUDA_SUCCESS);
    assert(cuCtxDestroy(ctx) == CUDA_SUCCESS);
    return write(1, "clean", 5) == 5 ? 0 : 1;
  }
  assert(argc == 2);
  int raw[2];
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, raw) == 0);
  int e = open(argv[0], O_RDONLY), m = open("/tmp", O_RDONLY | O_DIRECTORY);
  assert(e >= 0 && m >= 0);
  int ef = fcntl(e, F_DUPFD_CLOEXEC, 6);
  int mf = fcntl(m, F_DUPFD_CLOEXEC, 6);
  int kf = fcntl(m, F_DUPFD_CLOEXEC, 6);
  int cf = fcntl(raw[1], F_DUPFD_CLOEXEC, 6);
  close(e); close(m); close(raw[1]);
  assert(setenv("LUNAFLUX_UNRELATED_SENTINEL", "not-forwarded", 1) == 0);
  pid_t pid;
  assert(lf_process_spawn_approved(ef, cf, mf, kf, &pid) == 0);
  close(ef); close(mf); close(kf); close(cf);
  char result[5];
  assert(read(raw[0], result, 5) == 5 && memcmp(result, "clean", 5) == 0);
  close(raw[0]);
  int status;
  assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  puts("actual approved spawn: child CUDA, descriptors, release and unrelated-env exclusion passed");
  return 0;
}
