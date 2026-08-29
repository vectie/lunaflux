#ifndef LUNAFLUX_DEVICE_WORKER_ALLOC_FAKE_DEVICE_INTERNAL_H
#define LUNAFLUX_DEVICE_WORKER_ALLOC_FAKE_DEVICE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

typedef struct fake_context {
  int live;
  int children;
} fake_context;

typedef struct fake_allocation {
  fake_context *context;
  uint8_t *storage;
  size_t size;
  int live;
  int leases;
} fake_allocation;

typedef struct {
  fake_context *context;
  int live;
} fake_stream;

extern uint64_t lunaflux_device_worker_fake_launch_calls;
extern uint64_t lunaflux_device_worker_fake_sync_calls;
extern int lunaflux_device_worker_fake_fault;

void lunaflux_device_worker_fake_resource_opened_internal(void);
void lunaflux_device_worker_fake_resource_closed_internal(void);
int lunaflux_device_worker_fake_valid_region_internal(
  fake_context *context,
  fake_allocation *allocation,
  int64_t offset,
  int64_t count,
  int64_t alignment
);

#endif
