#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
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
int32_t lunaflux_process_inherited_try_write(
  uint8_t *, int32_t, int32_t
);
int32_t lunaflux_process_inherited_try_read(
  uint8_t *, int32_t, int32_t
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

static int probe_inherited_duplex(
  moonbit_bytes_t input,
  moonbit_bytes_t output
) {
  int read_sockets[2] = {-1, -1};
  int write_sockets[2] = {-1, -1};
  int saved_stdin = -1;
  int saved_stdout = -1;
  int result = 22;
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, read_sockets) != 0 ||
      socketpair(AF_UNIX, SOCK_STREAM, 0, write_sockets) != 0) {
    goto cleanup;
  }
  if (!set_nonblocking(read_sockets[0]) ||
      !set_nonblocking(read_sockets[1]) ||
      !set_nonblocking(write_sockets[0]) ||
      !set_nonblocking(write_sockets[1])) {
    result = 23;
    goto cleanup;
  }
  saved_stdin = dup(STDIN_FILENO);
  saved_stdout = dup(STDOUT_FILENO);
  if (saved_stdin < 0 || saved_stdout < 0 ||
      dup2(read_sockets[0], STDIN_FILENO) < 0 ||
      dup2(write_sockets[0], STDOUT_FILENO) < 0) {
    result = 24;
    goto cleanup;
  }
  if (lunaflux_process_inherited_try_read(input, 0, 4) != 0) {
    result = 25;
    goto cleanup;
  }
  if (lunaflux_process_inherited_try_write(output, 0, 4) != 4) {
    result = 26;
    goto cleanup;
  }
  uint8_t peer[4] = {0, 0, 0, 0};
  if (recv(write_sockets[1], peer, sizeof(peer), 0) != 4) {
    result = 27;
    goto cleanup;
  }
  for (int index = 0; index < 4; index += 1) {
    if (peer[index] != 7) {
      result = 28;
      goto cleanup;
    }
  }
  uint8_t first_half[2] = {8, 9};
  if (send(read_sockets[1], first_half, sizeof(first_half), 0) != 2 ||
      lunaflux_process_inherited_try_read(input, 0, 4) != 2 ||
      lunaflux_process_inherited_try_read(input, 2, 2) != 0) {
    result = 29;
    goto cleanup;
  }
  uint8_t second_half[2] = {10, 11};
  if (send(read_sockets[1], second_half, sizeof(second_half), 0) != 2 ||
      lunaflux_process_inherited_try_read(input, 2, 2) != 2 ||
      input[0] != 8 || input[1] != 9 || input[2] != 10 || input[3] != 11) {
    result = 30;
    goto cleanup;
  }
  (void)close(read_sockets[1]);
  read_sockets[1] = -1;
  struct pollfd inherited_close = {
    .fd = STDIN_FILENO,
    .events = POLLIN,
    .revents = 0,
  };
  if (poll(&inherited_close, 1, 1000) <= 0) {
    result = 31;
    goto cleanup;
  }
  if (lunaflux_process_inherited_try_read(input, 0, 1) !=
      -LF_PROCESS_CHANNEL_CLOSED) {
    result = 31;
    goto cleanup;
  }
  result = 0;

cleanup:
  if (saved_stdin >= 0) {
    (void)dup2(saved_stdin, STDIN_FILENO);
    (void)close(saved_stdin);
  }
  if (saved_stdout >= 0) {
    (void)dup2(saved_stdout, STDOUT_FILENO);
    (void)close(saved_stdout);
  }
  for (int index = 0; index < 2; index += 1) {
    if (read_sockets[index] >= 0) (void)close(read_sockets[index]);
    if (write_sockets[index] >= 0) (void)close(write_sockets[index]);
  }
  return result;
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
  if (lf_process_status_from_io_result(-1, ECONNRESET, 0) !=
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
  struct pollfd process_close = {
    .fd = sockets[0],
    .events = POLLIN,
    .revents = 0,
  };
  if (poll(&process_close, 1, 1000) <= 0) return 20;
  if (lunaflux_process_try_read(&process, input, 0, 1) !=
      -LF_PROCESS_CHANNEL_CLOSED) {
    return 20;
  }
  (void)close(sockets[0]);
  int64_t before = lunaflux_process_monotonic_now();
  int64_t after = lunaflux_process_monotonic_now();
  if (before < 0 || after < before) return 21;
  int inherited_result = probe_inherited_duplex(input, output);
  if (inherited_result != 0) return inherited_result;
  free_bytes(input);
  free_bytes(output);
  free_bytes(large);
  return 0;
}
