#include <stdint.h>
#include "process_status.h"

int32_t lunaflux_process_test_inherited_wait(int32_t mode);

int main(void) {
  if (lunaflux_process_test_inherited_wait(0) != LF_PROCESS_TIMEOUT) return 1;
  if (lunaflux_process_test_inherited_wait(1) != LF_PROCESS_OK) return 2;
  if (lunaflux_process_test_inherited_wait(2) != LF_PROCESS_OK) return 3;
  if (lunaflux_process_test_inherited_wait(3) != LF_PROCESS_OK) return 4;
  if (lunaflux_process_test_inherited_wait(4) != LF_PROCESS_TIMEOUT) return 5;
  return 0;
}
