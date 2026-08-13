#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include "process_handle.h"
#include "process_io.h"
#include "process_status.h"

int64_t lunaflux_process_monotonic_now(void);
int32_t lunaflux_process_try_write(
  lf_process *, uint8_t *, int32_t, int32_t
);
int32_t lunaflux_process_try_read(
  lf_process *, uint8_t *, int32_t, int32_t
);

static int set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static moonbit_bytes_t make_bytes(int32_t length, int value) {
  struct moonbit_object *header = calloc(
    1, sizeof(struct moonbit_object) + (size_t)length
  );
  if (header == NULL) return NULL;
  header->meta = (uint32_t)length;
  moonbit_bytes_t bytes = (moonbit_bytes_t)(header + 1);
  for (int32_t index = 0; index < length; index += 1) {
    bytes[index] = (uint8_t)value;
  }
  return bytes;
}

static void free_bytes(moonbit_bytes_t bytes) {
  if (bytes != NULL) free(Moonbit_object_header(bytes));
}

int main(void) {
  if (lf_process_status_from_io_result(-1, EINTR, 0) != LF_PROCESS_PENDING) {
    return 1;
  }
  if (lf_process_status_from_io_result(-1, EAGAIN, 1) != LF_PROCESS_PENDING) {
    return 2;
  }
  if (lf_process_status_from_io_result(-1, EPIPE, 1) !=
      LF_PROCESS_CHANNEL_CLOSED) {
    return 3;
  }
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return 4;
  if (!set_nonblocking(sockets[0]) || !set_nonblocking(sockets[1])) return 5;
  lf_process process = {
    .pid = -1,
    .fd = sockets[0],
    .reaped = 1,
    .closed = 0,
    .exit_kind = 0,
    .exit_code = 0,
  };
  moonbit_bytes_t input = make_bytes(4, 0);
  moonbit_bytes_t output = make_bytes(4, 7);
  if (input == NULL || output == NULL) return 6;
  if (lunaflux_process_try_read(&process, input, 0, 4) != 0) return 7;
  if (lunaflux_process_try_write(&process, output, 0, 4) != 4) return 8;
  uint8_t peer[4] = {0, 0, 0, 0};
  if (recv(sockets[1], peer, sizeof(peer), 0) != 4) return 9;
  for (int index = 0; index < 4; index += 1) {
    if (peer[index] != 7) return 10;
  }
  uint8_t source[2] = {1, 2};
  if (send(sockets[1], source, sizeof(source), 0) != 2) return 11;
  if (lunaflux_process_try_read(&process, input, 1, 3) != 2) return 12;
  if (input[0] != 0 || input[1] != 1 || input[2] != 2 || input[3] != 0) {
    return 13;
  }
  if (lunaflux_process_try_read(&process, input, -1, 1) !=
      -LF_PROCESS_FAILED) {
    return 14;
  }
  if (lunaflux_process_try_read(&process, input, 3, 2) !=
      -LF_PROCESS_FAILED) {
    return 15;
  }
  if (lunaflux_process_try_write(&process, output, 4, 1) !=
      -LF_PROCESS_FAILED) {
    return 16;
  }
  int send_buffer = 1024;
  (void)setsockopt(
    sockets[0], SOL_SOCKET, SO_SNDBUF, &send_buffer, sizeof(send_buffer)
  );
  moonbit_bytes_t large = make_bytes(65536, 3);
  if (large == NULL) return 17;
  int saw_short = 0;
  int saw_pending = 0;
  for (int attempt = 0; attempt < 10000; attempt += 1) {
    int32_t result = lunaflux_process_try_write(
      &process, large, 0, Moonbit_array_length(large)
    );
    if (result == 0) {
      saw_pending = 1;
      break;
    }
    if (result < 0) return 18;
    if (result < Moonbit_array_length(large)) saw_short = 1;
  }
  if (!saw_short || !saw_pending) return 19;
  (void)close(sockets[1]);
  if (lunaflux_process_try_read(&process, input, 0, 1) !=
      -LF_PROCESS_CHANNEL_CLOSED) {
    return 20;
  }
  (void)close(sockets[0]);
  int64_t before = lunaflux_process_monotonic_now();
  int64_t after = lunaflux_process_monotonic_now();
  if (before < 0 || after < before) return 21;
  free_bytes(input);
  free_bytes(output);
  free_bytes(large);
  return 0;
}
