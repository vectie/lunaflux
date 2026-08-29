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

typedef struct lf_inherited_drain lf_inherited_drain;
lf_inherited_drain *lunaflux_inherited_drain_open_fixed(void);
int32_t lunaflux_inherited_drain_open_status(lf_inherited_drain *);
int32_t lunaflux_inherited_drain_try_read(
  lf_inherited_drain *, uint8_t *, int32_t, int32_t
);
int32_t lunaflux_inherited_drain_peek(lf_inherited_drain *);
int32_t lunaflux_inherited_drain_try_write(
  lf_inherited_drain *, uint8_t *, int32_t, int32_t
);
int32_t lunaflux_inherited_drain_close(lf_inherited_drain *);
int32_t lunaflux_inherited_drain_is_closed(lf_inherited_drain *);

static void install_fixed_socket(int peer[1]) {
  int sockets[2] = {-1, -1};
  (void)close(5);
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  if (sockets[0] != 5) {
    assert(dup2(sockets[0], 5) == 5);
    assert(close(sockets[0]) == 0);
  }
  peer[0] = sockets[1];
}

int main(void) {
  (void)close(5);
  lf_inherited_drain *missing = lunaflux_inherited_drain_open_fixed();
  assert(lunaflux_inherited_drain_open_status(missing) == 1);
  assert(lunaflux_inherited_drain_close(missing) == 0);
  probe_free(missing);

  int pipes[2] = {-1, -1};
  assert(pipe(pipes) == 0);
  if (pipes[0] != 5) {
    assert(dup2(pipes[0], 5) == 5);
    assert(close(pipes[0]) == 0);
  }
  lf_inherited_drain *invalid = lunaflux_inherited_drain_open_fixed();
  assert(lunaflux_inherited_drain_open_status(invalid) == 2);
  assert(lunaflux_inherited_drain_close(invalid) == 0);
  assert(close(pipes[1]) == 0);
  (void)close(5);
  probe_free(invalid);

  int peer[1] = {-1};
  install_fixed_socket(peer);
  lf_inherited_drain *owner = lunaflux_inherited_drain_open_fixed();
  assert(lunaflux_inherited_drain_open_status(owner) == 0);
  assert(fcntl(5, F_GETFD, 0) < 0);
  moonbit_bytes_t input = probe_bytes(8);
  moonbit_bytes_t output = probe_bytes(8);
  const uint8_t command[8] = {'L', 'F', 'D', '1', 'D', 'R', 'N', '\n'};
  assert(send(peer[0], command, 3, 0) == 3);
  assert(lunaflux_inherited_drain_try_read(owner, input, 0, 8) == 3);
  assert(lunaflux_inherited_drain_try_read(owner, input, 3, 5) == 0);
  assert(send(peer[0], command + 3, 5, 0) == 5);
  assert(lunaflux_inherited_drain_try_read(owner, input, 3, 5) == 5);
  assert(lunaflux_inherited_drain_peek(owner) == 0);
  for (int index = 0; index < 8; index += 1) output[index] = command[index];
  assert(lunaflux_inherited_drain_try_write(owner, output, 0, 8) == 8);
  uint8_t received[8] = {0};
  assert(recv(peer[0], received, sizeof(received), 0) == 8);
  assert(send(peer[0], command, 1, 0) == 1);
  assert(lunaflux_inherited_drain_peek(owner) == 1);
  assert(lunaflux_inherited_drain_close(owner) == 0);
  assert(lunaflux_inherited_drain_is_closed(owner));
  assert(close(peer[0]) == 0);
  probe_free(input);
  probe_free(output);
  probe_free(owner);
  return 0;
}
