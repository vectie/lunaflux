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

#define LF_DRAIN_FIXED_FD 5
#define LF_DRAIN_OK 0
#define LF_DRAIN_UNAVAILABLE 1
#define LF_DRAIN_INVALID 2
#define LF_DRAIN_CLOSED 3

typedef struct {
  int fd;
  int status;
} lf_inherited_drain;

static void lf_inherited_drain_finalize(void *pointer) {
  lf_inherited_drain *owner = (lf_inherited_drain *)pointer;
  if (owner->fd >= 0) {
    (void)close(owner->fd);
    owner->fd = -1;
  }
}

static int lf_inherited_drain_validate(int fd) {
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
lf_inherited_drain *lunaflux_inherited_drain_open_fixed(void) {
  lf_inherited_drain *owner = (lf_inherited_drain *)moonbit_make_external_object(
    lf_inherited_drain_finalize, sizeof(lf_inherited_drain)
  );
  owner->fd = -1;
  owner->status = LF_DRAIN_UNAVAILABLE;
  if (fcntl(LF_DRAIN_FIXED_FD, F_GETFD, 0) < 0) return owner;
  if (!lf_inherited_drain_validate(LF_DRAIN_FIXED_FD)) {
    owner->status = LF_DRAIN_INVALID;
    return owner;
  }
  int moved = fcntl(LF_DRAIN_FIXED_FD, F_DUPFD_CLOEXEC, 6);
  if (moved < 0) {
    owner->status = LF_DRAIN_INVALID;
    return owner;
  }
  (void)close(LF_DRAIN_FIXED_FD);
  int flags = fcntl(moved, F_GETFL, 0);
  if (flags < 0 || fcntl(moved, F_SETFL, flags | O_NONBLOCK) != 0) {
    (void)close(moved);
    owner->status = LF_DRAIN_INVALID;
    return owner;
  }
  owner->fd = moved;
  owner->status = LF_DRAIN_OK;
  return owner;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_open_status(lf_inherited_drain *owner) {
  return owner == NULL ? LF_DRAIN_INVALID : owner->status;
}

static int32_t lf_drain_io_result(ssize_t count) {
  if (count > INT32_MAX) return -LF_DRAIN_INVALID;
  if (count >= 0) return (int32_t)count;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
    return -LF_DRAIN_CLOSED;
  }
  return -LF_DRAIN_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_try_read(
  lf_inherited_drain *owner,
  uint8_t *destination,
  int32_t offset,
  int32_t length
) {
  if (owner == NULL || owner->fd < 0 || destination == NULL || offset < 0 ||
      length <= 0 || offset > Moonbit_array_length(destination) - length) {
    return -LF_DRAIN_INVALID;
  }
  ssize_t count = recv(
    owner->fd, destination + offset, (size_t)length, 0
  );
  return count == 0 ? -LF_DRAIN_CLOSED : lf_drain_io_result(count);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_peek(lf_inherited_drain *owner) {
  if (owner == NULL || owner->fd < 0) return -LF_DRAIN_CLOSED;
  uint8_t byte = 0;
  ssize_t count = recv(owner->fd, &byte, 1, MSG_PEEK);
  if (count > 0) return 1;
  if (count == 0) return -LF_DRAIN_CLOSED;
  if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) return 0;
  return -LF_DRAIN_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_try_write(
  lf_inherited_drain *owner,
  uint8_t *source,
  int32_t offset,
  int32_t length
) {
  if (owner == NULL || owner->fd < 0 || source == NULL || offset < 0 ||
      length <= 0 || offset > Moonbit_array_length(source) - length) {
    return -LF_DRAIN_INVALID;
  }
#ifdef MSG_NOSIGNAL
  int flags = MSG_NOSIGNAL;
#else
  int flags = 0;
#endif
  return lf_drain_io_result(send(
    owner->fd, source + offset, (size_t)length, flags
  ));
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_close(lf_inherited_drain *owner) {
  if (owner == NULL) return LF_DRAIN_INVALID;
  if (owner->fd < 0) return LF_DRAIN_OK;
  int fd = owner->fd;
  owner->fd = -1;
  (void)shutdown(fd, SHUT_RDWR);
  return close(fd) == 0 ? LF_DRAIN_OK : LF_DRAIN_INVALID;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inherited_drain_is_closed(lf_inherited_drain *owner) {
  return owner != NULL && owner->fd < 0;
}
