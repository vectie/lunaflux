#include <fcntl.h>
#include <stdint.h>

int32_t lunaflux_process_test_expect_clean_eof(int32_t mode);

static int open_fd_count(void) {
  int count = 0;
  for (int fd = 0; fd < 64; fd += 1) {
    if (fcntl(fd, F_GETFD) >= 0) count += 1;
  }
  return count;
}

int main(void) {
  int before = open_fd_count();
  if (lunaflux_process_test_expect_clean_eof(0) != 0) return 1;
  if (lunaflux_process_test_expect_clean_eof(1) != 4) return 2;
  if (lunaflux_process_test_expect_clean_eof(2) != 4) return 3;
  if (lunaflux_process_test_expect_clean_eof(3) != 2) return 4;
  if (open_fd_count() != before) return 5;
  return 0;
}
