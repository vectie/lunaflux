#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <assert.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct {
  struct moonbit_object header;
  uint8_t payload[];
} lf_probe_object;

void *moonbit_make_external_object(
  void (*finalize)(void *),
  uint32_t payload_size
) {
  (void)finalize;
  lf_probe_object *object = (lf_probe_object *)calloc(
    1, sizeof(lf_probe_object) + payload_size
  );
  assert(object != NULL);
  return object->payload;
}

static moonbit_bytes_t probe_bytes(int32_t length) {
  lf_probe_object *object = (lf_probe_object *)calloc(
    1, sizeof(lf_probe_object) + (size_t)length
  );
  assert(object != NULL);
  object->header.meta = (uint32_t)length;
  return object->payload;
}

static void probe_free(void *payload) {
  if (payload != NULL) {
    free((uint8_t *)payload - offsetof(lf_probe_object, payload));
  }
}

typedef struct lf_inference_credential lf_inference_credential;
lf_inference_credential *lunaflux_inference_credential_open_fixed(void);
int32_t lunaflux_inference_credential_open_status(
  lf_inference_credential *
);
int32_t lunaflux_inference_credential_try_read(
  lf_inference_credential *, uint8_t *, int32_t, int32_t
);
int32_t lunaflux_inference_credential_peek(lf_inference_credential *);
int32_t lunaflux_inference_credential_close(lf_inference_credential *);
int32_t lunaflux_inference_credential_is_closed(lf_inference_credential *);

static void install_fixed_socket(int peer[1]) {
  int sockets[2] = {-1, -1};
  (void)close(6);
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  if (sockets[0] != 6) {
    assert(dup2(sockets[0], 6) == 6);
    assert(close(sockets[0]) == 0);
  }
  peer[0] = sockets[1];
}

int main(void) {
  (void)close(6);
  lf_inference_credential *missing =
    lunaflux_inference_credential_open_fixed();
  assert(lunaflux_inference_credential_open_status(missing) == 1);
  assert(lunaflux_inference_credential_close(missing) == 0);
  probe_free(missing);

  int pipes[2] = {-1, -1};
  assert(pipe(pipes) == 0);
  if (pipes[0] != 6) {
    assert(dup2(pipes[0], 6) == 6);
    assert(close(pipes[0]) == 0);
  }
  lf_inference_credential *invalid =
    lunaflux_inference_credential_open_fixed();
  assert(lunaflux_inference_credential_open_status(invalid) == 2);
  assert(fcntl(6, F_GETFD, 0) < 0);
  assert(lunaflux_inference_credential_close(invalid) == 0);
  assert(close(pipes[1]) == 0);
  probe_free(invalid);

  int peer[1] = {-1};
  install_fixed_socket(peer);
  lf_inference_credential *owner =
    lunaflux_inference_credential_open_fixed();
  assert(lunaflux_inference_credential_open_status(owner) == 0);
  assert(fcntl(6, F_GETFD, 0) < 0);
  moonbit_bytes_t input = probe_bytes(16);
  const uint8_t frame[15] = {
    'L', 'F', 'C', '1', 'K', 'E', 'Y', '\n', 3, 0, 0, 0, 'k', 'e', 'y'
  };
  assert(send(peer[0], frame, 5, 0) == 5);
  assert(lunaflux_inference_credential_try_read(owner, input, 0, 12) == 5);
  assert(send(peer[0], frame + 5, 10, 0) == 10);
  assert(lunaflux_inference_credential_try_read(owner, input, 5, 7) == 7);
  assert(lunaflux_inference_credential_try_read(owner, input, 12, 3) == 3);
  assert(lunaflux_inference_credential_peek(owner) == 0);
  assert(shutdown(peer[0], SHUT_WR) == 0);
  assert(lunaflux_inference_credential_peek(owner) == -3);
  assert(lunaflux_inference_credential_close(owner) == 0);
  assert(lunaflux_inference_credential_is_closed(owner));
  assert(close(peer[0]) == 0);
  probe_free(input);
  probe_free(owner);
  return 0;
}
