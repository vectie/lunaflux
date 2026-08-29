#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

#define LF_CREDENTIAL_FIXED_FD 6

typedef struct {
  int fd;
} lf_credential_peer;

static void lf_credential_peer_finalize(void *pointer) {
  lf_credential_peer *peer = (lf_credential_peer *)pointer;
  if (peer->fd >= 0) {
    (void)close(peer->fd);
    peer->fd = -1;
  }
}

static lf_credential_peer *lf_credential_peer_new(void) {
  lf_credential_peer *peer =
    (lf_credential_peer *)moonbit_make_external_object(
      lf_credential_peer_finalize, sizeof(lf_credential_peer)
    );
  peer->fd = -1;
  return peer;
}

MOONBIT_FFI_EXPORT
lf_credential_peer *lunaflux_inference_credential_fixture_install(
  int32_t mode
) {
  lf_credential_peer *peer = lf_credential_peer_new();
  (void)close(LF_CREDENTIAL_FIXED_FD);
  if (mode == 0) {
    int sockets[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) return peer;
    if (sockets[0] != LF_CREDENTIAL_FIXED_FD &&
        dup2(sockets[0], LF_CREDENTIAL_FIXED_FD) < 0) {
      (void)close(sockets[0]);
      (void)close(sockets[1]);
      return peer;
    }
    if (sockets[0] != LF_CREDENTIAL_FIXED_FD) (void)close(sockets[0]);
    int flags = fcntl(sockets[1], F_GETFL, 0);
    if (flags < 0 || fcntl(sockets[1], F_SETFL, flags | O_NONBLOCK) != 0) {
      (void)close(LF_CREDENTIAL_FIXED_FD);
      (void)close(sockets[1]);
      return peer;
    }
    peer->fd = sockets[1];
  } else {
    int pipes[2] = {-1, -1};
    if (pipe(pipes) != 0) return peer;
    if (pipes[0] == LF_CREDENTIAL_FIXED_FD ||
        dup2(pipes[0], LF_CREDENTIAL_FIXED_FD) >= 0) {
      peer->fd = pipes[1];
    } else {
      (void)close(pipes[1]);
    }
    if (pipes[0] != LF_CREDENTIAL_FIXED_FD) (void)close(pipes[0]);
  }
  return peer;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_fixture_send(
  lf_credential_peer *peer,
  uint8_t *source,
  int32_t offset,
  int32_t length
) {
  if (peer == NULL || peer->fd < 0 || source == NULL || offset < 0 ||
      length <= 0 || offset > Moonbit_array_length(source) - length) return -1;
  ssize_t count = send(peer->fd, source + offset, (size_t)length, 0);
  return count > INT32_MAX ? -1 : (int32_t)count;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_fixture_close_write(
  lf_credential_peer *peer
) {
  return peer == NULL || peer->fd < 0 ? -1 : shutdown(peer->fd, SHUT_WR);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_fixture_close(
  lf_credential_peer *peer
) {
  if (peer == NULL || peer->fd < 0) return 0;
  int fd = peer->fd;
  peer->fd = -1;
  return close(fd);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_fixture_clear_fixed(void) {
  return close(LF_CREDENTIAL_FIXED_FD) == 0 || errno == EBADF ? 0 : -1;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_fixture_fixed_closed(void) {
  return fcntl(LF_CREDENTIAL_FIXED_FD, F_GETFD, 0) < 0 && errno == EBADF;
}
