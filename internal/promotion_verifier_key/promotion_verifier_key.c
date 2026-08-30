#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define LF_PROMOTION_KEY_FIXED_FD 7
#define LF_PROMOTION_KEY_OK 0
#define LF_PROMOTION_KEY_UNAVAILABLE 1
#define LF_PROMOTION_KEY_INVALID 2
#define LF_PROMOTION_KEY_CLOSED 3

typedef struct {
  int fd;
  int status;
} lf_promotion_verifier_key;

static void lf_promotion_verifier_key_finalize(void *pointer) {
  lf_promotion_verifier_key *owner =
    (lf_promotion_verifier_key *)pointer;
  if (owner->fd >= 0) {
    (void)close(owner->fd);
    owner->fd = -1;
  }
}

static int lf_promotion_verifier_key_validate(int fd) {
  struct stat info;
  int socket_type = 0;
  socklen_t type_length = sizeof(socket_type);
  struct sockaddr_un local;
  struct sockaddr_un peer;
  socklen_t local_length = sizeof(local);
  socklen_t peer_length = sizeof(peer);
  return fstat(fd, &info) == 0 && S_ISSOCK(info.st_mode) &&
    getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &type_length) == 0 &&
    type_length == sizeof(socket_type) && socket_type == SOCK_STREAM &&
    getsockname(fd, (struct sockaddr *)&local, &local_length) == 0 &&
    local.sun_family == AF_UNIX &&
    getpeername(fd, (struct sockaddr *)&peer, &peer_length) == 0 &&
    peer.sun_family == AF_UNIX;
}

MOONBIT_FFI_EXPORT
lf_promotion_verifier_key *lunaflux_promotion_verifier_key_open_fixed(void) {
  lf_promotion_verifier_key *owner =
    (lf_promotion_verifier_key *)moonbit_make_external_object(
      lf_promotion_verifier_key_finalize,
      sizeof(lf_promotion_verifier_key)
  );
  owner->fd = -1;
  owner->status = LF_PROMOTION_KEY_UNAVAILABLE;
  int descriptor_flags = fcntl(LF_PROMOTION_KEY_FIXED_FD, F_GETFD, 0);
  if (descriptor_flags < 0) return owner;
  if ((descriptor_flags & FD_CLOEXEC) != 0) return owner;
  if (!lf_promotion_verifier_key_validate(LF_PROMOTION_KEY_FIXED_FD)) {
    owner->status = LF_PROMOTION_KEY_INVALID;
    (void)close(LF_PROMOTION_KEY_FIXED_FD);
    return owner;
  }
  int moved = fcntl(LF_PROMOTION_KEY_FIXED_FD, F_DUPFD_CLOEXEC, 8);
  if (moved < 0) {
    owner->status = LF_PROMOTION_KEY_INVALID;
    (void)close(LF_PROMOTION_KEY_FIXED_FD);
    return owner;
  }
  (void)close(LF_PROMOTION_KEY_FIXED_FD);
  int flags = fcntl(moved, F_GETFL, 0);
  if (flags < 0 || fcntl(moved, F_SETFL, flags | O_NONBLOCK) != 0) {
    (void)close(moved);
    owner->status = LF_PROMOTION_KEY_INVALID;
    return owner;
  }
  owner->fd = moved;
  owner->status = LF_PROMOTION_KEY_OK;
  return owner;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_promotion_verifier_key_open_status(
  lf_promotion_verifier_key *owner
) {
  return owner == NULL ? LF_PROMOTION_KEY_INVALID : owner->status;
}

static int32_t lf_promotion_verifier_key_io_result(ssize_t count) {
  if (count > INT32_MAX) return -LF_PROMOTION_KEY_INVALID;
  if (count >= 0) return (int32_t)count;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
    return -LF_PROMOTION_KEY_CLOSED;
  }
  return -LF_PROMOTION_KEY_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_promotion_verifier_key_try_read(
  lf_promotion_verifier_key *owner,
  uint8_t *destination,
  int32_t offset,
  int32_t length
) {
  if (owner == NULL || owner->fd < 0 || destination == NULL || offset < 0 ||
      length <= 0 || offset > Moonbit_array_length(destination) - length) {
    return -LF_PROMOTION_KEY_INVALID;
  }
  ssize_t count = recv(owner->fd, destination + offset, (size_t)length, 0);
  return count == 0 ? -LF_PROMOTION_KEY_CLOSED :
    lf_promotion_verifier_key_io_result(count);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_promotion_verifier_key_peek(
  lf_promotion_verifier_key *owner
) {
  if (owner == NULL || owner->fd < 0) return -LF_PROMOTION_KEY_CLOSED;
  uint8_t byte = 0;
  ssize_t count = recv(owner->fd, &byte, 1, MSG_PEEK);
  if (count > 0) return 1;
  if (count == 0) return -LF_PROMOTION_KEY_CLOSED;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  return -LF_PROMOTION_KEY_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_promotion_verifier_key_close(
  lf_promotion_verifier_key *owner
) {
  if (owner == NULL) return LF_PROMOTION_KEY_INVALID;
  if (owner->fd < 0) return LF_PROMOTION_KEY_OK;
  int fd = owner->fd;
  owner->fd = -1;
  (void)shutdown(fd, SHUT_RDWR);
  return close(fd) == 0 ? LF_PROMOTION_KEY_OK : LF_PROMOTION_KEY_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_promotion_verifier_key_is_closed(
  lf_promotion_verifier_key *owner
) {
  return owner != NULL && owner->fd < 0;
}
