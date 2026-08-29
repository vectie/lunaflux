#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

static int fixture_peer = -1;
static int fixture_saved_stdin = -1;
static int fixture_saved_stdout = -1;
static volatile uint64_t fixture_progress_mask;
static volatile uint64_t fixture_progress_count;

static int write_exact(int fd, const uint8_t *source, int32_t length) {
  int32_t cursor = 0;
  while (cursor < length) {
    ssize_t count = send(
      fd, source + cursor, (size_t)(length - cursor), MSG_NOSIGNAL
    );
    if (count <= 0) return -1;
    cursor += (int32_t)count;
  }
  return 0;
}

static int read_exact(int fd, uint8_t *destination, int32_t length) {
  int32_t cursor = 0;
  while (cursor < length) {
    ssize_t count = recv(fd, destination + cursor, (size_t)(length - cursor), 0);
    if (count <= 0) return -1;
    cursor += (int32_t)count;
  }
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_begin(void) {
  if (fixture_peer >= 0) return -1;
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return -1;
  fixture_saved_stdin = dup(STDIN_FILENO);
  fixture_saved_stdout = dup(STDOUT_FILENO);
  if (fixture_saved_stdin < 0 || fixture_saved_stdout < 0 ||
      dup2(sockets[0], STDIN_FILENO) < 0 ||
      dup2(sockets[0], STDOUT_FILENO) < 0) {
    if (fixture_saved_stdin >= 0) close(fixture_saved_stdin);
    if (fixture_saved_stdout >= 0) close(fixture_saved_stdout);
    close(sockets[0]);
    close(sockets[1]);
    fixture_saved_stdin = -1;
    fixture_saved_stdout = -1;
    return -1;
  }
  close(sockets[0]);
  fixture_peer = sockets[1];
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_send(
  moonbit_bytes_t source,
  int32_t length
) {
  if (fixture_peer < 0 || source == NULL || length <= 0 ||
      length > Moonbit_array_length(source)) return -1;
  uint8_t prefix[4] = {
    (uint8_t)length,
    (uint8_t)(length >> 8),
    (uint8_t)(length >> 16),
    (uint8_t)(length >> 24),
  };
  if (write_exact(fixture_peer, prefix, 4) != 0 ||
      write_exact(fixture_peer, source, length) != 0) return -1;
  fixture_progress_mask |= 1U;
  fixture_progress_count += 1U;
  return 0;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_receive(moonbit_bytes_t destination) {
  if (fixture_peer < 0 || destination == NULL) return -1;
  uint8_t prefix[4] = {0, 0, 0, 0};
  if (read_exact(fixture_peer, prefix, 4) != 0) return -1;
  int32_t length = (int32_t)prefix[0] |
    ((int32_t)prefix[1] << 8) |
    ((int32_t)prefix[2] << 16) |
    ((int32_t)prefix[3] << 24);
  if (length <= 0 || length > Moonbit_array_length(destination) ||
      read_exact(fixture_peer, destination, length) != 0) return -1;
  fixture_progress_mask |= 2U;
  fixture_progress_count += 1U;
  return length;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_shutdown_write(void) {
  return fixture_peer >= 0 && shutdown(fixture_peer, SHUT_WR) == 0 ? 0 : -1;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_finish(void) {
  int status = 0;
  if (fixture_peer >= 0 && close(fixture_peer) != 0) status = -1;
  fixture_peer = -1;
  if (fixture_saved_stdin < 0 ||
      dup2(fixture_saved_stdin, STDIN_FILENO) < 0) status = -1;
  if (fixture_saved_stdout < 0 ||
      dup2(fixture_saved_stdout, STDOUT_FILENO) < 0) status = -1;
  if (fixture_saved_stdin >= 0) close(fixture_saved_stdin);
  if (fixture_saved_stdout >= 0) close(fixture_saved_stdout);
  fixture_saved_stdin = -1;
  fixture_saved_stdout = -1;
  return status;
}

MOONBIT_FFI_EXPORT
void lunaflux_rank_child_fixture_reset_evidence(void) {
  fixture_progress_mask = 0U;
  fixture_progress_count = 0U;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_rank_child_fixture_check_evidence(
  uint32_t expected_mask,
  uint64_t minimum_count
) {
  return fixture_progress_mask == expected_mask &&
    fixture_progress_count >= minimum_count ? 1 : 0;
}
