/* CPU fake-driver regression. No checkpoint, GPU workload or production gate. */
typedef struct lf_copy_probe {
  int copies;
  int fail_copy;
  int fail_record;
  int not_ready;
  int synchronizations;
  int destroyed;
} lf_copy_probe;

static lf_copy_probe *lf_active_copy_probe;

static CUresult lf_copy_probe_current(CUcontext context) {
  (void)context; return 0;
}
static CUresult lf_copy_probe_create_event(CUevent *event, uint32_t flags) {
  if (flags != 2) return 1;
  *event = (void *)(uintptr_t)1;
  return 0;
}
static CUresult lf_copy_probe_destroy_event(CUevent event) {
  (void)event; ++lf_active_copy_probe->destroyed; return 0;
}
static CUresult lf_copy_probe_record(CUevent event, CUstream stream) {
  (void)event; (void)stream;
  return lf_active_copy_probe->fail_record ? 1 : 0;
}
static CUresult lf_copy_probe_query(CUevent event) {
  (void)event;
  if (lf_active_copy_probe->not_ready) {
    lf_active_copy_probe->not_ready = 0;
    return CUDA_ERROR_NOT_READY;
  }
  return 0;
}
static CUresult lf_copy_probe_sync(CUstream stream) {
  (void)stream; ++lf_active_copy_probe->synchronizations; return 0;
}
static CUresult lf_copy_probe_memcpy(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream stream) {
  (void)stream;
  ++lf_active_copy_probe->copies;
  if (lf_active_copy_probe->copies == lf_active_copy_probe->fail_copy) return 1;
  memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, bytes);
  return 0;
}
static void lf_copy_probe_finalizer(void *object) { (void)object; }

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_device_copy(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; ++cycle) {
    lf_copy_probe state = {0};
    lf_active_copy_probe = &state;
    lf_cuda_api api = {0};
    api.cuCtxSetCurrent = lf_copy_probe_current;
    api.cuMemcpyDtoDAsync = lf_copy_probe_memcpy;
    api.cuEventCreate = lf_copy_probe_create_event;
    api.cuEventDestroy = lf_copy_probe_destroy_event;
    api.cuEventRecord = lf_copy_probe_record;
    api.cuEventQuery = lf_copy_probe_query;
    api.cuStreamSynchronize = lf_copy_probe_sync;
    lf_context context = {0}, other_context = {0};
    context.api = &api;
    atomic_init(&context.state, LF_RESOURCE_LIVE);
    atomic_init(&context.active_operations, 0);
    lf_child *stream = moonbit_make_external_object(lf_copy_probe_finalizer, sizeof(*stream));
    memset(stream, 0, sizeof(*stream));
    stream->context = &context;
    atomic_init(&stream->state, LF_RESOURCE_LIVE);
    atomic_init(&stream->active_operations, 0);
    uint8_t data[32], backup[32];
    memset(data, 0x2a, sizeof(data));
    memset(backup, 0, sizeof(backup));
    lf_allocation **sources = (lf_allocation **)moonbit_make_extern_ref_array_raw(1);
    lf_allocation **destinations = (lf_allocation **)moonbit_make_extern_ref_array_raw(1);
    lf_allocation *source = moonbit_make_external_object(lf_copy_probe_finalizer, sizeof(*source));
    lf_allocation *destination = moonbit_make_external_object(lf_copy_probe_finalizer, sizeof(*destination));
    memset(source, 0, sizeof(*source)); memset(destination, 0, sizeof(*destination));
    source->context = &context; destination->context = &context;
    source->handle = (uintptr_t)data; destination->handle = (uintptr_t)backup;
    source->size = 32; destination->size = 32;
    atomic_init(&source->state, LF_RESOURCE_LIVE);
    atomic_init(&source->active_operations, 0);
    atomic_init(&destination->state, LF_RESOURCE_LIVE);
    atomic_init(&destination->active_operations, 0);
    sources[0] = source; destinations[0] = destination;
    int64_t *offsets = moonbit_make_int64_array(1, 4);
    int64_t *sizes = moonbit_make_int64_array(1, 20);
    int32_t status = 0, result = 0;
    lf_device_copy *copy = lunaflux_cuda_device_copy_create(
      &context, stream, sources, destinations, offsets, offsets, sizes, &status);
    if (status != LF_OK || atomic_load(&source->active_operations) != 1 ||
        atomic_load(&destination->active_operations) != 1 ||
        atomic_load(&stream->active_operations) != 1) result = 101;
    if (!result && lf_begin_close(&source->state, &source->active_operations) != LF_BUSY) result = 102;
    if (!result && lunaflux_cuda_device_copy_submit(copy, 0) != LF_OK) result = 103;
    if (!result && (backup[3] != 0 || backup[4] != 0x2a || backup[23] != 0x2a || backup[24] != 0)) result = 104;
    if (!result && (lunaflux_cuda_device_copy_submit(copy, 1) != LF_BUSY ||
        lunaflux_cuda_device_copy_close(copy) != LF_BUSY)) result = 105;
    state.not_ready = 1;
    if (!result && (lunaflux_cuda_device_copy_retire(copy, 0) != 0 ||
        lunaflux_cuda_device_copy_retire(copy, 0) != 1)) result = 106;
    memset(data+4, 0x7b, 20);
    if (!result && (lunaflux_cuda_device_copy_submit(copy, 1) != LF_OK ||
        lunaflux_cuda_device_copy_retire(copy, 1) != 1 || data[4] != 0x2a ||
        data[23] != 0x2a || state.synchronizations != 1)) result = 107;
    state.fail_copy = state.copies+1;
    if (!result && (lunaflux_cuda_device_copy_submit(copy, 0) != LF_DRIVER_FAILURE ||
        lunaflux_cuda_device_copy_close(copy) != LF_BUSY ||
        lunaflux_cuda_device_copy_retire(copy, 0) != LF_DRIVER_FAILURE ||
        lunaflux_cuda_device_copy_retire(copy, 1) != 1)) result = 108;
    state.fail_record = 1;
    if (!result && (lunaflux_cuda_device_copy_submit(copy, 0) != LF_DRIVER_FAILURE ||
        lunaflux_cuda_device_copy_retire(copy, 1) != 1)) result = 109;
    state.fail_record = 0;
    if (!result && (lunaflux_cuda_device_copy_close(copy) != LF_OK ||
        lunaflux_cuda_device_copy_close(copy) != LF_OK ||
        atomic_load(&source->active_operations) != 0 ||
        atomic_load(&destination->active_operations) != 0 ||
        atomic_load(&stream->active_operations) != 0 || state.destroyed != 1)) result = 110;
    moonbit_decref(copy);
    /* Invalid preparations must retain no partial leases after explicit close. */
    sizes[0] = 29;
    copy = lunaflux_cuda_device_copy_create(
      &context, stream, sources, destinations, offsets, offsets, sizes, &status);
    if (!result && (status != LF_INVALID_ARGUMENT ||
        lunaflux_cuda_device_copy_close(copy) != LF_OK)) result = 201;
    moonbit_decref(copy);
    sizes[0] = 20;
    destination->context = &other_context;
    copy = lunaflux_cuda_device_copy_create(
      &context, stream, sources, destinations, offsets, offsets, sizes, &status);
    if (!result && (status != LF_INVALID_ARGUMENT ||
        lunaflux_cuda_device_copy_close(copy) != LF_OK)) result = 202;
    moonbit_decref(copy);
    destination->context = &context;
    destination->handle = source->handle+10;
    copy = lunaflux_cuda_device_copy_create(
      &context, stream, sources, destinations, offsets, offsets, sizes, &status);
    if (!result && (status != LF_INVALID_ARGUMENT ||
        lunaflux_cuda_device_copy_close(copy) != LF_OK)) result = 203;
    moonbit_decref(copy);
    if (!result && (atomic_load(&source->active_operations) ||
        atomic_load(&destination->active_operations) ||
        atomic_load(&stream->active_operations))) result = 204;
    moonbit_decref(sizes); moonbit_decref(offsets);
    moonbit_decref(destinations); moonbit_decref(sources);
    moonbit_decref(destination); moonbit_decref(source); moonbit_decref(stream);
    lf_active_copy_probe = NULL;
    if (result) return result;
  }
  return LF_OK;
}
