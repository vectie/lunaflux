#define _GNU_SOURCE 1
#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#include <stdatomic.h>
#include <pthread.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>

/* Offline injection library, never linked into the engine. Bounded activity
 * storage, no stack sampling/module snapshots, no CUDA synchronization. Raw
 * records are streamed to disk; analysis/conversion runs after model retirement.
 * CUDA loads InitializeInjection at cuInit, not in the HTTP parent before fork.
 * CUPTI 13.0.1 ABI; deliberately not an unversioned cross-toolkit binary. */
#define BUFFERS 8
#define BUFFER_BYTES (1024 * 1024)
#define WINDOWS 4
static _Alignas(8) uint8_t storage[BUFFERS][BUFFER_BYTES];
static atomic_int occupied[BUFFERS];
static atomic_ulong refused;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *records;
static uint64_t begins[WINDOWS], ends[WINDOWS];
static unsigned windows;
static uint64_t dropped, invalid, count;
static atomic_uint_fast64_t errors;
static int control_fd = -1;
static pthread_t controller;
static char root[512], endpoint[108];
static atomic_int shutting_down;
static int initialized;
static int enabled;
static int sealed;
static const CUpti_ActivityKind kinds[] = {
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
  CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_DRIVER,
  CUPTI_ACTIVITY_KIND_RUNTIME
};
static int checked(CUptiResult r);

/* Omit polling/context binding BEFORE records are allocated. The observed
 * native rank generated 857563 context binds and 857257 event queries in one
 * failed window. Filtering emitted text after CUPTI buffering cannot fix that.
 * Launches, synchronization/wait calls and every device activity remain enabled.
 * Consequently this collector cannot attribute CPU cost to individual polls. */
static void omit_poll_activity(void) {
  const CUpti_CallbackId driver[] = {
    CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent,
    CUPTI_DRIVER_TRACE_CBID_cuEventQuery,
    CUPTI_DRIVER_TRACE_CBID_cuStreamQuery,
    CUPTI_DRIVER_TRACE_CBID_cuStreamQuery_ptsz
  };
  const CUpti_CallbackId runtime[] = {
    CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020,
    CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000
  };
  for (unsigned i=0;i<sizeof(driver)/sizeof(driver[0]);++i)
    checked(cuptiActivityEnableDriverApi(driver[i],0));
  for (unsigned i=0;i<sizeof(runtime)/sizeof(runtime[0]);++i)
    checked(cuptiActivityEnableRuntimeApi(runtime[i],0));
}

static int checked(CUptiResult r) {
  if (r == CUPTI_SUCCESS) return 1;
  const char *s = NULL;
  cuptiGetResultString(r, &s);
  fprintf(stderr, "offline activity collector: %s\n", s ? s : "unknown error");
  ++errors;
  return 0;
}

static void CUPTIAPI request_buffer(uint8_t **buffer, size_t *size, size_t *max) {
  *max = 0; *size = 0; *buffer = NULL;
  for (int i = 0; i < BUFFERS; ++i) {
    int free = 0;
    if (atomic_compare_exchange_strong(&occupied[i], &free, 1)) {
      *buffer = storage[i]; *size = BUFFER_BYTES; return;
    }
  }
  atomic_fetch_add(&refused, 1);
}

static unsigned window_for(uint64_t start, uint64_t end) {
  if (!start || !end || end < start) { ++invalid; return 0; }
  for (unsigned i = 0; i < windows; ++i)
    if (start >= begins[i] && (!ends[i] || start < ends[i])) return i + 1;
  return 0;
}

static void emit(const char *kind, uint64_t start, uint64_t end,
                 uint32_t stream, uint32_t correlation, uint64_t node,
                 uint32_t graph, const char *name, uint32_t thread,
                 uint64_t bytes, int gx, int gy, int gz, int bx, int by, int bz) {
  unsigned w = window_for(start, end);
  if (!w) return;
  if (!name || strnlen(name, 8192) == 8192 || strpbrk(name, "\t\r\n")) {
    ++invalid; return;
  }
  if (fprintf(records, "%u\t%s\t%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%"
              PRIu64 "\t%u\t%u\t%" PRIu64 "\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n",
              w, kind, start, end, stream, correlation, node, graph, thread,
              bytes, gx, gy, gz, bx, by, bz, name) < 0) ++errors;
  ++count;
}

static void CUPTIAPI complete_buffer(CUcontext ctx, uint32_t stream,
                                     uint8_t *buffer, size_t size, size_t valid) {
  (void)size;
  pthread_mutex_lock(&lock);
  fprintf(records, "BUFFER size=%zu valid=%zu stream=%u\n", size, valid, stream);
  CUpti_Activity *a = NULL;
  CUptiResult r = CUPTI_ERROR_MAX_LIMIT_REACHED;
  while (valid && (r = cuptiActivityGetNextRecord(buffer, valid, &a)) == CUPTI_SUCCESS) {
    switch (a->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        const CUpti_ActivityKernel10 *k = (const void *)a;
        emit("K", k->start, k->end, k->streamId, k->correlationId,
             k->graphNodeId, k->graphId, k->name, 0, 0,
             k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const CUpti_ActivityMemcpy6 *m = (const void *)a;
        emit("C", m->start, m->end, m->streamId, m->correlationId,
             m->graphNodeId, m->graphId, "memcpy", 0, m->bytes, 0,0,0,0,0,0);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        const CUpti_ActivityMemset4 *m = (const void *)a;
        emit("Z", m->start, m->end, m->streamId, m->correlationId,
             m->graphNodeId, m->graphId, "memset", 0, m->bytes, 0,0,0,0,0,0);
        break;
      }
      case CUPTI_ACTIVITY_KIND_DRIVER:
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const CUpti_ActivityAPI *api = (const void *)a;
        const char *name = NULL;
        if (checked(cuptiGetCallbackName(a->kind == CUPTI_ACTIVITY_KIND_DRIVER
                         ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API,
                         api->cbid, &name)))
          emit(a->kind == CUPTI_ACTIVITY_KIND_DRIVER ? "D" : "R",
               api->start, api->end, 0, api->correlationId, 0, 0, name,
               api->threadId, 0, 0,0,0,0,0,0);
        break;
      }
      default: break;
    }
  }
  if (r != CUPTI_ERROR_MAX_LIMIT_REACHED) checked(r);
  size_t lost = 0;
  checked(cuptiActivityGetNumDroppedRecords(ctx, stream, &lost));
  dropped += lost;
  pthread_mutex_unlock(&lock);
  for (int i = 0; i < BUFFERS; ++i)
    if (buffer == storage[i]) { atomic_store(&occupied[i], 0); return; }
  atomic_fetch_add(&refused, 1);
}

static uint64_t stamp(void) {
  uint64_t value = 0; checked(cuptiGetTimestamp(&value)); return value;
}

static void status(FILE *out, const char *event) {
  struct timespec before, after;
  clock_gettime(CLOCK_REALTIME, &before);
  uint64_t gpu_clock = stamp();
  clock_gettime(CLOCK_REALTIME, &after);
  fprintf(out, "%s pid=%d windows=%u records=%" PRIu64
          " dropped=%" PRIu64 " invalid=%" PRIu64 " errors=%" PRIu64
          " refused=%lu timestamp=%" PRIu64 " realtime_before=%" PRIu64
          " realtime_after=%" PRIu64 "\n", event, getpid(), windows, count,
          dropped, invalid, (uint64_t)atomic_load(&errors), atomic_load(&refused), gpu_clock,
          (uint64_t)before.tv_sec * 1000000000 + before.tv_nsec,
          (uint64_t)after.tv_sec * 1000000000 + after.tv_nsec);
  for (unsigned i = 0; i < windows; ++i)
    fprintf(out, "window=%u begin=%" PRIu64 " end=%" PRIu64 "\n",
            i + 1, begins[i], ends[i]);
  fflush(out);
}

static void *control(void *unused) {
  (void)unused;
  while (!atomic_load(&shutting_down)) {
    struct sockaddr_un peer;
    socklen_t len = sizeof(peer);
    char command[16];
    ssize_t n = recvfrom(control_fd, command, sizeof(command)-1, 0,
                         (struct sockaddr *)&peer, &len);
    if (n < 0) continue;
    command[n] = 0;
    if (!strcmp(command, "start")) {
      if (sealed) { ++errors; continue; }
      // Initialize injection before CUDA/graph creation, but record only after
      // model readiness. Startup's incomplete API records must not occupy the
      // bounded window collector. Keep graph IDs from already-created graphs.
      if (!enabled) {
        for (unsigned i=0; i<sizeof(kinds)/sizeof(kinds[0]); ++i)
          checked(cuptiActivityEnable(kinds[i]));
        omit_poll_activity();
        enabled = 1;
      }
      pthread_mutex_lock(&lock);
      if (windows == WINDOWS || (windows && !ends[windows-1])) ++errors;
      else { begins[windows] = stamp(); ++windows; }
      pthread_mutex_unlock(&lock);
    } else if (!strcmp(command, "stop")) {
      pthread_mutex_lock(&lock);
      if (!windows || ends[windows-1]) ++errors;
      else ends[windows-1] = stamp();
      pthread_mutex_unlock(&lock);
      // Drain completed records only; no cuCtxSynchronize. Later completed
      // records retain their timestamp-based window membership until close.
      checked(cuptiActivityFlushAll(0));
    } else if (!strcmp(command, "seal")) {
      // The external owner sends this only after the HTTP response has drained.
      // Python workers may use _exit; completeness must not depend on atexit.
      checked(cuptiActivityFlushPeriod(0));
      if (enabled)
        for (unsigned i=0; i<sizeof(kinds)/sizeof(kinds[0]); ++i)
          checked(cuptiActivityDisable(kinds[i]));
      enabled = 0;
      checked(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
      pthread_mutex_lock(&lock);
      if (sealed || !windows || !ends[windows-1]) ++errors;
      if (!sealed) { status(records,"FINAL"); sealed = 1; }
      pthread_mutex_unlock(&lock);
    } else if (strcmp(command, "status")) ++errors;
    char reply[1024];
    FILE *text = fmemopen(reply, sizeof(reply), "w");
    pthread_mutex_lock(&lock);
    status(text, command); fflush(records);
    long bytes = ftell(text);
    fclose(text);
    pthread_mutex_unlock(&lock);
    if (bytes > 0 && bytes < (long)sizeof(reply))
      sendto(control_fd, reply, (size_t)bytes, 0, (struct sockaddr *)&peer, len);
  }
  return NULL;
}

static void finish(void) {
  if (!initialized) return;
  atomic_store(&shutting_down, 1);
  pthread_join(controller, NULL);
  checked(cuptiActivityFlushPeriod(0));
  if (enabled)
    for (unsigned i=0; i<sizeof(kinds)/sizeof(kinds[0]); ++i)
      checked(cuptiActivityDisable(kinds[i]));
  checked(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  pthread_mutex_lock(&lock);
  if (!sealed) status(records, "FINAL");
  fclose(records); records = NULL;
  pthread_mutex_unlock(&lock);
  close(control_fd); unlink(endpoint);
}

int InitializeInjection(void) {
  if (initialized) return 1;
  const char *path = getenv("LUNAFLUX_ACTIVITY_ROOT");
  if (!path || strlen(path) >= sizeof(root)) return 0;
  strcpy(root, path);
  unsigned version = 0;
  if (!checked(cuptiGetVersion(&version)) || version != CUPTI_API_VERSION) return 0;
  size_t size = BUFFER_BYTES, pool = 4, pre = 2, n = sizeof(size);
  uint8_t per_thread = 0; size_t one = sizeof(per_thread);
  if (!checked(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &n, &size)) ||
      !checked(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &n, &pool)) ||
      !checked(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &n, &pre)) ||
      !checked(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_PER_THREAD_ACTIVITY_BUFFER, &one, &per_thread))) return 0;
  char file[600];
  snprintf(file, sizeof(file), "%s/activity-%d.tsv", root, getpid());
  records = fopen(file, "wx");
  if (!records) return 0;
  fprintf(records, "SCHEMA1 pid=%d cupti=%u host_bytes=%d device_pool=%zu device_buffer_bytes=%zu api_scope=launch-wait-copy-without-poll-or-context-bind\n",
          getpid(), version, BUFFERS*BUFFER_BYTES, pool, size);
  if (!checked(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer))) return 0;
  // Drain completed window records promptly. Activity itself remains disabled
  // during startup. No GPU synchronization is requested.
  if (!checked(cuptiActivityFlushPeriod(10))) return 0;
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  if (snprintf(endpoint, sizeof(endpoint), "%s/control-%d.sock", root, getpid()) >= (int)sizeof(endpoint)) return 0;
  strcpy(address.sun_path, endpoint);
  control_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (control_fd < 0 || bind(control_fd, (struct sockaddr *)&address, sizeof(address))) return 0;
  struct timeval timeout = {.tv_usec = 100000};
  if (setsockopt(control_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout))) return 0;
  if (pthread_create(&controller, NULL, control, NULL)) return 0;
  initialized = 1;
  atexit(finish);
  return 1;
}
