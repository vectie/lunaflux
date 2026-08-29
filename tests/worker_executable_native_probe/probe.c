#define _DARWIN_C_SOURCE 1
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>

#include "../../internal/approved_fs_capability/approved_fs_capability.h"

extern int32_t lunaflux_worker_executable_close(
  lf_approved_executable *owner
);

MOONBIT_FFI_EXPORT
int32_t lunaflux_worker_executable_test_close_while_active(
  lf_approved_executable *owner
) {
  if (owner == NULL) return 1;
  uint32_t state = atomic_load_explicit(&owner->state, memory_order_acquire);
  for (;;) {
    if ((state & LF_APPROVED_STATE_CLOSING) != 0) return 4;
    if (state == LF_APPROVED_ACTIVE_MAX) return 7;
    if (atomic_compare_exchange_weak_explicit(
          &owner->state, &state, state + 1,
          memory_order_acq_rel, memory_order_acquire
        )) break;
  }
  int32_t status = lunaflux_worker_executable_close(owner);
  (void)atomic_fetch_sub_explicit(&owner->state, 1, memory_order_release);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_worker_executable_test_link(
  moonbit_bytes_t source,
  moonbit_bytes_t destination
) {
  if (source == NULL || destination == NULL) return 1;
  int32_t source_length = Moonbit_array_length(source);
  int32_t destination_length = Moonbit_array_length(destination);
  if (source_length <= 0 || destination_length <= 0 || source_length > 4096 ||
      destination_length > 4096) return 1;
  char source_path[4097];
  char destination_path[4097];
  if (memchr(source, '\0', (size_t)source_length) != NULL ||
      memchr(destination, '\0', (size_t)destination_length) != NULL) return 1;
  memcpy(source_path, source, (size_t)source_length);
  memcpy(destination_path, destination, (size_t)destination_length);
  source_path[source_length] = '\0';
  destination_path[destination_length] = '\0';
  return link(source_path, destination_path) == 0 ? 0 : 6;
}
