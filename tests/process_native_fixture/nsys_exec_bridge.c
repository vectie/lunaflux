#define _GNU_SOURCE 1
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>

/* Offline profiler interposer, never linked into LunaFlux. Preserve only the
 * observed Nsight startup settings across the production fexecve boundary.
 * Snapshot before fork; the interposed child performs no allocation, lookup,
 * dlsym or stdio. No request data, CUDA calls or compiler policy lives here. */
static const char *const keys[] = {
  "LD_PRELOAD", "TMPDIR", "TRACE_START_IMMEDIATELY", "NSYS_AGENT_TMP_DIR",
  "QUADD_AGENT_PID", "QUADD_INJECTION_CONFIG", "__GL_CONSTANT_FRAME_RATE_HINT",
  "NSYS_CUDA_DRIVER_TYPE", "NSYS_TSC_SUPPORT", "NSYS_CUDA_GPU_TIME_CONVERSION",
  "NSYS_PROFILING_SESSION_ID", "NSYS_SESSION_TEMP_DIR",
  "NSYS_TARGET_STD_REDIRECT_ACTION", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
  "NSYS_CONTROL_CUPTI_FEATURES", "CUPTI_PROFILE_MODE", "QUADD_CUDA_CONFIG",
  /* NCCL INFO must never write into the worker's framed stdout transport. */
  "NCCL_DEBUG_FILE", "NCCL_PROFILER_PLUGIN", "LUNAFLUX_NCCL_PROGRESS_ROOT"
};
static char snapshot[sizeof(keys) / sizeof(keys[0])][PATH_MAX + 1];
static size_t count;
static int snapshot_error;

__attribute__((constructor)) static void snapshot_profiler(void) {
  for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
    const char *value = getenv(keys[i]);
    if (!value) continue;
    size_t k = strlen(keys[i]), v = strlen(value);
    if (k + v + 2 > sizeof(snapshot[0])) { snapshot_error = E2BIG; return; }
    memcpy(snapshot[count], keys[i], k);
    snapshot[count][k] = '=';
    memcpy(snapshot[count] + k + 1, value, v + 1);
    ++count;
  }
}

int fexecve(int fd, char *const argv[], char *const envp[]) {
  char *combined[128];
  size_t n = 0;
  if (snapshot_error) { errno = snapshot_error; return -1; }
  for (size_t i = 0; envp[i]; ++i) {
    if (n + count + 1 >= sizeof(combined) / sizeof(combined[0])) {
      errno = E2BIG; return -1;
    }
    combined[n++] = envp[i];
  }
  for (size_t i = 0; i < count; ++i) combined[n++] = snapshot[i];
  combined[n] = NULL;
  return (int)syscall(SYS_execveat, fd, "", argv, combined, AT_EMPTY_PATH);
}
