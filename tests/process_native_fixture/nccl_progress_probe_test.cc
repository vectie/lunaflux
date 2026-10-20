#include "nccl_progress_probe.cc"
#include <assert.h>
#include <thread>

int main(int argc, char **argv) {
  assert(argc == 2);
  setenv("LUNAFLUX_NCCL_PROGRESS_ROOT", argv[1], 1);
  char path[1024];
  snprintf(path, sizeof(path), "%s/progress.control", argv[1]);
  int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
  assert(fd >= 0 && ftruncate(fd, 4096) == 0);
  auto *control = static_cast<uint32_t *>(mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  close(fd);
  assert(control != MAP_FAILED);
  void *context = nullptr;
  int mask = 0;
  assert(init(&context, 0, &mask, "test", 2, 2, 0, nullptr) == ncclSuccess);
  assert(mask == (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileProxyCtrl));
  ncclProfilerEventDescr_v5_t d{};
  d.type = ncclProfileColl; d.rank = 0;
  d.coll.seqNumber = 91; d.coll.count = 327680;
  d.coll.nChannels = 32; d.coll.algo = "RING"; d.coll.proto = "SIMPLE";
  void *h = nullptr;
  assert(start(context, &h, &d) == ncclSuccess && h == nullptr);
  __atomic_store_n(control, 1, __ATOMIC_RELEASE);
  assert(start(context, &h, &d) == ncclSuccess && h != nullptr);
  auto *coll = static_cast<Event *>(h);
  assert(coll->seq == 91 && coll->count == 327680 && coll->phase == 1);
  d = {}; d.type = ncclProfileProxyOp; d.parentObj = h;
  d.proxyOp.channelId = 3; d.proxyOp.isSend = 1; d.proxyOp.nSteps = 2;
  void *op = nullptr;
  assert(start(context, &op, &d) == ncclSuccess && op != nullptr);
  d = {}; d.type = ncclProfileProxyStep; d.parentObj = op; d.proxyStep.step = 1;
  void *step = nullptr;
  assert(start(context, &step, &d) == ncclSuccess && step != nullptr);
  auto *e = static_cast<Event *>(step);
  assert(e->parent == static_cast<Event *>(op)->id && e->seq == 91 && e->send == 1 && e->channel == 3);
  __atomic_store_n(control, 0, __ATOMIC_RELEASE);
  std::thread peer([&] {
    ncclProfilerEventStateArgs_v5_t args{}; args.proxyStep.transSize = 10240;
    assert(state(step, ncclProfilerProxyStepSendGPUWait, &args) == ncclSuccess);
    assert(state(step, ncclProfilerProxyStepSendWait, &args) == ncclSuccess);
    assert(stop(step) == ncclSuccess);
  });
  peer.join();
  assert(e->states[8] && e->states[9] && e->end >= e->states[9] && e->count == 10240);
  assert(stop(op) == ncclSuccess && stop(h) == ncclSuccess);
  auto *ctx = static_cast<Context *>(context);
  __atomic_store_n(control, 2, __ATOMIC_RELEASE);
  d = {}; d.type = ncclProfileProxyCtrl;
  void *ctrl = nullptr;
  assert(start(context, &ctrl, &d) == ncclSuccess && ctrl != nullptr);
  ncclProfilerEventStateArgs_v5_t args{};
  assert(state(ctrl, ncclProfilerProxyCtrlAppend, &args) == ncclSuccess);
  args.proxyCtrl.appendedProxyOps = 1088;
  assert(state(ctrl, ncclProfilerProxyCtrlAppendEnd, &args) == ncclSuccess);
  assert(stop(ctrl) == ncclSuccess);
  auto *control_event = static_cast<Event *>(ctrl);
  assert(control_event->phase == 2 && control_event->parent == 0 && control_event->count == 1088);
  assert(control_event->states[17] && control_event->states[18] >= control_event->states[17]);
  assert(control_event->cpu_states[18] >= control_event->cpu_states[17]);
  d = {}; d.type = ncclProfileProxyStep; d.parentObj = op;
  ctx->next = capacity;
  assert(start(context, &h, &d) == ncclSuccess && h == nullptr && ctx->dropped == 1);
  // Do not print uninitialized synthetic overflow slots in this fixture.
  ctx->next = 4;
  assert(finish(context) == ncclSuccess);
  munmap(control, 4096);
  puts("bounded phase/parent/state/overflow/release checks pass");
}
