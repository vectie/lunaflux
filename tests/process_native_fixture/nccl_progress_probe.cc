// Offline, bounded NCCL v5 profiler. Never linked into the engine.
// Build against the unmodified v2.31.2-1 example ABI headers. All event storage
// is allocated at init; callbacks only timestamp retained records. Comparison,
// experiment phase control and report parsing belong to external .mbtx tools.
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include "err.h"
#include "profiler.h"
#include <nvtx3/nvToolsExt.h>

static uint64_t now_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}

#ifdef PROGRESS_CONTROL
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  char path[1024];
  if (snprintf(path, sizeof(path), "%s/progress.control", argv[1]) >= int(sizeof(path))) return 2;
  if (!strcmp(argv[2], "init")) {
    int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return 3;
    int ok = ftruncate(fd, 4096);
    close(fd);
    return ok == 0 ? 0 : 4;
  }
  if (strlen(argv[2]) != 1 || argv[2][0] < '0' || argv[2][0] > '3') return 2;
  int fd = open(path, O_RDWR);
  if (fd < 0) return 3;
  void *p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return 4;
  __atomic_store_n(static_cast<uint32_t *>(p), uint32_t(argv[2][0] - '0'), __ATOMIC_RELEASE);
  printf("phase=%s monotonic_ns=%llu\n", argv[2], (unsigned long long)now_ns());
  munmap(p, 4096);
  return 0;
}
#else
struct Context;
struct Event {
  Context *ctx;
  uint64_t id, parent, type, begin, end, seq, count, states[32];
  uint64_t cpu_states[32], cpu_end;
  int phase, rank, channel, send, step, steps, chunk, tid;
  char algo[16], proto[16];
};
static constexpr uint64_t capacity = 65536;
struct Context {
  Event *events;
  uint32_t *control;
  uint64_t next, dropped, colls[4], controls[4];
  int fd, rank;
};

static ncclResult_t init(void **out, uint64_t, int *mask, const char *,
                         int, int, int rank, ncclDebugLogger_t) {
  *out = nullptr;
  *mask = 0;
  const char *root = getenv("LUNAFLUX_NCCL_PROGRESS_ROOT");
  if (!root) return ncclSuccess;
  char path[1024];
  if (snprintf(path, sizeof(path), "%s/progress.control", root) >= int(sizeof(path))) return ncclSystemError;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return ncclSystemError;
  void *control = mmap(nullptr, 4096, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (control == MAP_FAILED) return ncclSystemError;
  auto *ctx = new (std::nothrow) Context{};
  if (!ctx) { munmap(control, 4096); return ncclSystemError; }
  ctx->events = static_cast<Event *>(calloc(capacity, sizeof(Event)));
  ctx->control = static_cast<uint32_t *>(control);
  ctx->rank = rank;
  if (snprintf(path, sizeof(path), "%s/progress-rank%d-%d.tsv", root, rank, getpid()) >= int(sizeof(path))) {
    free(ctx->events); munmap(control, 4096); delete ctx; return ncclSystemError;
  }
  ctx->fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (!ctx->events || ctx->fd < 0) {
    if (ctx->fd >= 0) close(ctx->fd);
    free(ctx->events); munmap(control, 4096); delete ctx; return ncclSystemError;
  }
  nvtxMarkA("LunaFlux external NCCL progress observer initialized");
  // No KernelCh instrumentation: observe host proxy states, not extra GPU stores.
  *mask = ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileProxyCtrl;
  *out = ctx;
  return ncclSuccess;
}

static ncclResult_t start(void *context, void **handle, ncclProfilerEventDescr_v5_t *d) {
  *handle = nullptr;
  auto *ctx = static_cast<Context *>(context);
  if (!ctx) return ncclSuccess;
  int phase = __atomic_load_n(ctx->control, __ATOMIC_ACQUIRE);
  // A captured collective's API parent can predate the observed replay. Its
  // absent handle is allowed; actual Coll -> ProxyOp -> ProxyStep links persist.
  if (!phase || phase > 3 ||
      !(d->type == ncclProfileColl || d->type == ncclProfileProxyOp || d->type == ncclProfileProxyStep || d->type == ncclProfileProxyCtrl))
    return ncclSuccess;
  // A fixed first-chunk prefix per phase prevents cold report flushing from
  // consuming the warm window. Children of skipped callbacks are skipped too.
  if (d->type == ncclProfileColl) {
    if (__atomic_fetch_add(&ctx->colls[phase], 1, __ATOMIC_RELAXED) >= 64)
      return ncclSuccess;
  } else if (d->type == ncclProfileProxyCtrl) {
    if (__atomic_fetch_add(&ctx->controls[phase], 1, __ATOMIC_RELAXED) >= 4096)
      return ncclSuccess;
  } else if (!d->parentObj) return ncclSuccess;
  uint64_t id = __atomic_fetch_add(&ctx->next, 1, __ATOMIC_RELAXED);
  if (id >= capacity) {
    __atomic_fetch_add(&ctx->dropped, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
  }
  Event *e = &ctx->events[id];
  e->ctx = ctx; e->id = id + 1; e->type = d->type;
  e->phase = phase; e->rank = d->rank;
  e->channel = -1; e->send = -1; e->step = -1;
  e->tid = int(syscall(SYS_gettid));
  e->begin = now_ns();
  if (d->parentObj) {
    auto *parent = static_cast<Event *>(d->parentObj);
    e->parent = parent->id;
    e->seq = parent->seq;
    e->channel = parent->channel;
    e->send = parent->send;
  }
  if (d->type == ncclProfileColl) {
    e->seq = d->coll.seqNumber; e->count = d->coll.count;
    e->channel = d->coll.nChannels;
    snprintf(e->algo, sizeof(e->algo), "%s", d->coll.algo ? d->coll.algo : "");
    snprintf(e->proto, sizeof(e->proto), "%s", d->coll.proto ? d->coll.proto : "");
    char mark[160];
    snprintf(mark, sizeof(mark), "LF_NCCL phase=%d seq=%llu mono=%llu", phase,
             (unsigned long long)e->seq, (unsigned long long)e->begin);
    nvtxMarkA(mark);
  } else if (d->type == ncclProfileProxyOp) {
    e->channel = d->proxyOp.channelId; e->send = d->proxyOp.isSend;
    e->steps = d->proxyOp.nSteps; e->chunk = d->proxyOp.chunkSize;
  } else if (d->type == ncclProfileProxyStep) {
    e->step = d->proxyStep.step;
  }
  *handle = e;
  return ncclSuccess;
}

static ncclResult_t stop(void *handle) {
  if (handle) {
    struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    auto *e = static_cast<Event *>(handle);
    e->cpu_end = uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
    __atomic_store_n(&e->end, now_ns(), __ATOMIC_RELEASE);
  }
  return ncclSuccess;
}
static ncclResult_t state(void *handle, ncclProfilerEventState_v5_t s,
                          ncclProfilerEventStateArgs_v5_t *args) {
  if (!handle || int(s) < 0 || int(s) >= 32) return ncclSuccess;
  auto *e = static_cast<Event *>(handle);
  uint64_t zero = 0;
  if (__atomic_compare_exchange_n(&e->states[int(s)], &zero, now_ns(), false, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    e->cpu_states[int(s)] = uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
  }
  if (args && e->type == ncclProfileProxyStep)
    __atomic_store_n(&e->count, uint64_t(args->proxyStep.transSize), __ATOMIC_RELAXED);
  if (args && e->type == ncclProfileProxyCtrl)
    __atomic_store_n(&e->count, uint64_t(args->proxyCtrl.appendedProxyOps), __ATOMIC_RELAXED);
  return ncclSuccess;
}
static ncclResult_t finish(void *context) {
  auto *ctx = static_cast<Context *>(context);
  if (!ctx) return ncclSuccess;
  FILE *out = fdopen(ctx->fd, "w");
  if (!out) return ncclSystemError;
  uint64_t n = __atomic_load_n(&ctx->next, __ATOMIC_ACQUIRE);
  fprintf(out, "# capacity=%llu attempted=%llu dropped=%llu record_bytes=%zu\n",
          (unsigned long long)capacity, (unsigned long long)n,
          (unsigned long long)__atomic_load_n(&ctx->dropped, __ATOMIC_RELAXED), sizeof(Event));
  fprintf(out, "id\tparent\ttype\tphase\trank\tseq\tchannel\tsend\tstep\tsteps\tchunk\tcount\ttid\tbegin\tend\talgo\tproto");
  for (int s = 0; s < 32; ++s) fprintf(out, "\ts%d", s);
  fprintf(out, "\tcpu_end");
  for (int s = 0; s < 32; ++s) fprintf(out, "\tcpu_s%d", s);
  fputc('\n', out);
  for (uint64_t i = 0; i < n && i < capacity; ++i) {
    Event *e = &ctx->events[i];
    fprintf(out, "%llu\t%llu\t%llu\t%d\t%d\t%llu\t%d\t%d\t%d\t%d\t%d\t%llu\t%d\t%llu\t%llu\t%s\t%s",
            (unsigned long long)e->id, (unsigned long long)e->parent, (unsigned long long)e->type,
            e->phase, e->rank, (unsigned long long)e->seq, e->channel, e->send, e->step,
            e->steps, e->chunk, (unsigned long long)e->count, e->tid,
            (unsigned long long)e->begin, (unsigned long long)e->end, e->algo, e->proto);
    for (int s = 0; s < 32; ++s) fprintf(out, "\t%llu", (unsigned long long)e->states[s]);
    fprintf(out, "\t%llu", (unsigned long long)e->cpu_end);
    for (int s = 0; s < 32; ++s) fprintf(out, "\t%llu", (unsigned long long)e->cpu_states[s]);
    fputc('\n', out);
  }
  int result = fclose(out);
  munmap(ctx->control, 4096);
  free(ctx->events);
  delete ctx;
  return result == 0 ? ncclSuccess : ncclSystemError;
}
extern "C" {
__attribute__((visibility("default"))) ncclProfiler_v5_t ncclProfiler_v5 = {
  "LunaFlux-offline-progress", init, start, stop, state, finish
};
}
#endif
