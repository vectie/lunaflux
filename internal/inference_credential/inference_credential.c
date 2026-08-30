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

#define LF_CREDENTIAL_FIXED_FD 6
#define LF_CREDENTIAL_OK 0
#define LF_CREDENTIAL_UNAVAILABLE 1
#define LF_CREDENTIAL_INVALID 2
#define LF_CREDENTIAL_CLOSED 3

typedef struct {
  int fd;
  int status;
} lf_inference_credential;

static void lf_inference_credential_finalize(void *pointer) {
  lf_inference_credential *owner = (lf_inference_credential *)pointer;
  if (owner->fd >= 0) {
    (void)close(owner->fd);
    owner->fd = -1;
  }
}

static int lf_inference_credential_validate(int fd) {
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
lf_inference_credential *lunaflux_inference_credential_open_fixed(void) {
  lf_inference_credential *owner =
    (lf_inference_credential *)moonbit_make_external_object(
      lf_inference_credential_finalize, sizeof(lf_inference_credential)
  );
  owner->fd = -1;
  owner->status = LF_CREDENTIAL_UNAVAILABLE;
  int descriptor_flags = fcntl(LF_CREDENTIAL_FIXED_FD, F_GETFD, 0);
  if (descriptor_flags < 0) return owner;
  if ((descriptor_flags & FD_CLOEXEC) != 0) return owner;
  if (!lf_inference_credential_validate(LF_CREDENTIAL_FIXED_FD)) {
    owner->status = LF_CREDENTIAL_INVALID;
    (void)close(LF_CREDENTIAL_FIXED_FD);
    return owner;
  }
  int moved = fcntl(LF_CREDENTIAL_FIXED_FD, F_DUPFD_CLOEXEC, 7);
  if (moved < 0) {
    owner->status = LF_CREDENTIAL_INVALID;
    (void)close(LF_CREDENTIAL_FIXED_FD);
    return owner;
  }
  (void)close(LF_CREDENTIAL_FIXED_FD);
  int flags = fcntl(moved, F_GETFL, 0);
  if (flags < 0 || fcntl(moved, F_SETFL, flags | O_NONBLOCK) != 0) {
    (void)close(moved);
    owner->status = LF_CREDENTIAL_INVALID;
    return owner;
  }
  owner->fd = moved;
  owner->status = LF_CREDENTIAL_OK;
  return owner;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_open_status(
  lf_inference_credential *owner
) {
  return owner == NULL ? LF_CREDENTIAL_INVALID : owner->status;
}

static int32_t lf_credential_io_result(ssize_t count) {
  if (count > INT32_MAX) return -LF_CREDENTIAL_INVALID;
  if (count >= 0) return (int32_t)count;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
    return -LF_CREDENTIAL_CLOSED;
  }
  return -LF_CREDENTIAL_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_try_read(
  lf_inference_credential *owner,
  uint8_t *destination,
  int32_t offset,
  int32_t length
) {
  if (owner == NULL || owner->fd < 0 || destination == NULL || offset < 0 ||
      length <= 0 || offset > Moonbit_array_length(destination) - length) {
    return -LF_CREDENTIAL_INVALID;
  }
  ssize_t count = recv(owner->fd, destination + offset, (size_t)length, 0);
  return count == 0 ? -LF_CREDENTIAL_CLOSED :
    lf_credential_io_result(count);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_peek(lf_inference_credential *owner) {
  if (owner == NULL || owner->fd < 0) return -LF_CREDENTIAL_CLOSED;
  uint8_t byte = 0;
  ssize_t count = recv(owner->fd, &byte, 1, MSG_PEEK);
  if (count > 0) return 1;
  if (count == 0) return -LF_CREDENTIAL_CLOSED;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  return -LF_CREDENTIAL_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_close(lf_inference_credential *owner) {
  if (owner == NULL) return LF_CREDENTIAL_INVALID;
  if (owner->fd < 0) return LF_CREDENTIAL_OK;
  int fd = owner->fd;
  owner->fd = -1;
  (void)shutdown(fd, SHUT_RDWR);
  return close(fd) == 0 ? LF_CREDENTIAL_OK : LF_CREDENTIAL_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inference_credential_is_closed(
  lf_inference_credential *owner
) {
  return owner != NULL && owner->fd < 0;
}
