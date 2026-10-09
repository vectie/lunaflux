/* Included in fake_device.c, sharing its allocation leases/resource counters.
 * Copies complete on the second poll; host read and lane reuse before that are
 * rejected. All resources are acquired at startup, never in enqueue/poll. */
typedef struct {
  fake_allocation *allocation;
  uint8_t *host;
  size_t bytes;
  int phase;
  int polled;
} serial_transfer;

static void serial_transfer_finalize(void *raw) {
  serial_transfer *pool = raw;
  if (pool->allocation != NULL || pool->host != NULL) abort();
}

MOONBIT_FFI_EXPORT
void *lunaflux_serial_test_streaming_create(void *context, void *allocation,
    int64_t bytes, int32_t lanes, int32_t regions, int32_t *status) {
  serial_transfer *pool = moonbit_make_external_object(
      serial_transfer_finalize, sizeof(serial_transfer));
  memset(pool, 0, sizeof(*pool));
  fake_allocation *buffer = allocation;
  *status = -2;
  if (buffer == NULL || !buffer->live || buffer->context != context ||
      bytes <= 0 || (uint64_t)bytes > SIZE_MAX || lanes != 1 || regions != 1)
    return pool;
  pool->host = calloc((size_t)bytes, 1);
  if (pool->host == NULL) { *status = -6; return pool; }
  moonbit_incref(buffer);
  buffer->leases += 1;
  pool->allocation = buffer;
  pool->bytes = (size_t)bytes;
  opened();
  *status = 0;
  return pool;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_close(serial_transfer *pool) {
  if (pool->allocation == NULL) return 0;
  if (pool->phase == 1 || (pool->phase == 2 && pool->polled < 2)) return -4;
  pool->allocation->leases -= 1;
  moonbit_decref(pool->allocation);
  pool->allocation = NULL;
  free(pool->host);
  pool->host = NULL;
  closed();
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_enqueue(serial_transfer *pool,
    int32_t lane, int64_t host, int64_t device, int64_t bytes, int32_t direction) {
  if (pool->allocation == NULL || lane != 0 || pool->phase != 0 ||
      host < 0 || device < 0 || bytes <= 0 ||
      (uint64_t)host > pool->bytes || (uint64_t)bytes > pool->bytes - host ||
      (uint64_t)device > pool->allocation->size ||
      (uint64_t)bytes > pool->allocation->size - device) return -2;
  uint8_t *gpu = pool->allocation->storage + device;
  uint8_t *cpu = pool->host + host;
  if (direction == 1) memcpy(gpu, cpu, (size_t)bytes);
  else if (direction == 0) memcpy(cpu, gpu, (size_t)bytes);
  else return -2;
  pool->phase = 1;
  pool->polled = 0;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_record(serial_transfer *pool, int32_t lane) {
  if (pool->allocation == NULL || lane != 0 || pool->phase != 1) return -2;
  pool->phase = 2;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_poll(serial_transfer *pool, int32_t lane) {
  if (pool->allocation == NULL || lane != 0 || pool->phase != 2) return -2;
  if (pool->polled < 2) pool->polled += 1;
  if (pool->polled < 2) return 0;
  return 1;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_drain(serial_transfer *pool, int32_t lane) {
  if (pool->allocation == NULL) return 0;
  if (lane != 0) return -2;
  pool->polled = 2;
  pool->phase = 0;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_reset(serial_transfer *pool, int32_t lane) {
  if (pool->allocation == NULL || lane != 0 || pool->phase != 2 || pool->polled < 2)
    return -2;
  pool->phase = 0;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_serial_test_streaming_host_copy(serial_transfer *pool,
    uint8_t *buffer, int64_t host, int32_t offset, int32_t bytes,
    int32_t direction) {
  if (pool->allocation == NULL || host < 0 || offset < 0 || bytes < 0 ||
      (uint64_t)host > pool->bytes || (uint64_t)bytes > pool->bytes - host ||
      offset > Moonbit_array_length(buffer) ||
      bytes > Moonbit_array_length(buffer) - offset) return -2;
  if (direction == 1 && pool->phase == 0) {
    memcpy(pool->host + host, buffer + offset, (size_t)bytes);
  } else if (direction == 0 && pool->phase == 2 && pool->polled >= 2) {
    memcpy(buffer + offset, pool->host + host, (size_t)bytes);
  } else return -4;
  return 0;
}
