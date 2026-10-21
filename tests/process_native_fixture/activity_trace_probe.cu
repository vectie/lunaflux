#define _GNU_SOURCE 1
extern "C" {
#include "process_approved_spawn.h"
}
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

__global__ void activity_fixture(unsigned *out, unsigned value) {
  unsigned x = value;
  for (unsigned i = 0; i < 100000; ++i) x = x * 1664525u + 1013904223u;
  out[threadIdx.x] = x;
}

static void control(const char *cmd) {
  struct sockaddr_un a = {}, b = {};
  a.sun_family = b.sun_family = AF_UNIX;
  snprintf(a.sun_path, sizeof(a.sun_path), "/tmp/lf-fixture-%d.sock", getpid());
  snprintf(b.sun_path, sizeof(b.sun_path), "%s/control-%d.sock", getenv("LUNAFLUX_ACTIVITY_ROOT"), getpid());
  int fd = socket(AF_UNIX, SOCK_DGRAM, 0); assert(fd >= 0);
  assert(bind(fd, (sockaddr *)&a, sizeof(a)) == 0);
  struct timeval timeout = {15,0};
  assert(setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)) == 0);
  assert(sendto(fd, cmd, strlen(cmd), 0, (sockaddr *)&b, sizeof(b)) > 0);
  char response[2048]; ssize_t n = recv(fd,response,sizeof(response),0); assert(n > 0);
  assert(write(2,response,(size_t)n) == n);
  close(fd); unlink(a.sun_path);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    struct stat a, b;
    assert(fstat(3,&a) == 0 && S_ISDIR(a.st_mode));
    assert(fstat(4,&b) == 0 && S_ISDIR(b.st_mode));
    assert(getenv("LUNAFLUX_UNRELATED_SENTINEL") == NULL);
    unsigned *p, *q; cudaStream_t s, t;
    assert(cudaMalloc(&p,1024) == cudaSuccess);
    assert(cudaMalloc(&q,1024) == cudaSuccess);
    assert(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking) == cudaSuccess);
    assert(cudaStreamCreateWithFlags(&t,cudaStreamNonBlocking) == cudaSuccess);
    cudaGraph_t graph; cudaGraphExec_t exec;
    // Capture BEFORE observation begins, exactly as the serving owner does.
    assert(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal) == cudaSuccess);
    activity_fixture<<<1,256,0,s>>>(p,3);
    assert(cudaMemsetAsync(p,0,1024,s) == cudaSuccess);
    assert(cudaStreamEndCapture(s,&graph) == cudaSuccess);
    assert(cudaGraphInstantiate(&exec,graph,0) == cudaSuccess);
    CUcontext context;
    assert(cuCtxGetCurrent(&context) == CUDA_SUCCESS);
    CUevent completed;
    assert(cuEventCreate(&completed,CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS);
    assert(cuEventRecord(completed,(CUstream)s) == CUDA_SUCCESS);
    assert(cuEventSynchronize(completed) == CUDA_SUCCESS);
    // Deliberately exceed the entire host pool in API records before capture.
    // They must drain and be discarded by scope, not exhaust retained buffers.
    for (int i=0; i<1000000; ++i) {
      unsigned flags;
      assert(cudaStreamGetFlags(s,&flags) == cudaSuccess);
    }
    for (int window = 0; window < 2; ++window) {
      control("start");
      // Reproduce the measured native flood INSIDE the window, rather than
      // only a harmless pre-window API burst. These records must be excluded
      // before allocation; graph/eager device coverage must remain complete.
      for (int i=0; i<1000000; ++i) {
        assert(cuCtxSetCurrent(context) == CUDA_SUCCESS);
        assert(cuEventQuery(completed) == CUDA_SUCCESS);
      }
      for (int i = 0; i < 4; ++i) {
        assert(cudaGraphLaunch(exec,s) == cudaSuccess);
        activity_fixture<<<1,256,0,t>>>(q,7);
      }
      assert(cudaDeviceSynchronize() == cudaSuccess);
      control("stop");
    }
    assert(cudaGraphExecDestroy(exec) == cudaSuccess);
    assert(cuEventDestroy(completed) == CUDA_SUCCESS);
    assert(cudaGraphDestroy(graph) == cudaSuccess);
    assert(cudaStreamDestroy(s) == cudaSuccess && cudaStreamDestroy(t) == cudaSuccess);
    assert(cudaFree(p) == cudaSuccess && cudaFree(q) == cudaSuccess);
    assert(cudaDeviceReset() == cudaSuccess);
    control("seal");
    assert(write(1,"clean",5) == 5);
    _exit(0); // Match Python multiprocess retirement; no libc atexit callback.
  }
  assert(argc == 2);
  int pair[2]; assert(socketpair(AF_UNIX,SOCK_STREAM,0,pair) == 0);
  int e=open(argv[0],O_RDONLY),m=open("/tmp",O_RDONLY|O_DIRECTORY);
  assert(e>=0 && m>=0);
  int ef=fcntl(e,F_DUPFD_CLOEXEC,6),mf=fcntl(m,F_DUPFD_CLOEXEC,6);
  int kf=fcntl(m,F_DUPFD_CLOEXEC,6),cf=fcntl(pair[1],F_DUPFD_CLOEXEC,6);
  close(e); close(m); close(pair[1]);
  assert(setenv("LUNAFLUX_UNRELATED_SENTINEL","exclude",1) == 0);
  pid_t pid; assert(lf_process_spawn_approved(ef,cf,mf,kf,&pid) == 0);
  close(ef);close(mf);close(kf);close(cf);
  char result[5]; assert(read(pair[0],result,5) == 5 && !memcmp(result,"clean",5));
  close(pair[0]); int status;
  assert(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
  puts("approved child: two windows of eight kernels, four graph nodes each, explicit release");
}
