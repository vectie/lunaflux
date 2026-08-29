#include <stdint.h>
#include <stdlib.h>

static uint32_t lunaflux_rank_wire_execution_mask;
static uint64_t lunaflux_rank_wire_execution_count;

void lunaflux_rank_wire_execution_reset(void) {
  lunaflux_rank_wire_execution_mask = 0U;
  lunaflux_rank_wire_execution_count = 0U;
}

void lunaflux_rank_wire_execution_mark(int32_t bit) {
  if (bit <= 0 || bit > 8) {
    abort();
  }
  lunaflux_rank_wire_execution_mask |= (uint32_t)bit;
  ++lunaflux_rank_wire_execution_count;
}

int32_t lunaflux_rank_wire_execution_check(
  uint32_t expected_mask,
  uint64_t minimum_count
) {
  return lunaflux_rank_wire_execution_mask == expected_mask &&
      lunaflux_rank_wire_execution_count >= minimum_count
    ? 1
    : 0;
}
