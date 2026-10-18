#include <assert.h>
#include <stdio.h>
#include "provider.c"

/* Execute the production provider without the MoonBit allocator so ASan/LSan
 * see every native allocation. Only the byte-array length header is modeled. */
static uint8_t *bytes_new(size_t length) {
  struct moonbit_object *header = calloc(1, sizeof(*header) + length);
  assert(header != NULL && length <= INT32_MAX);
  header->meta = (uint32_t)length;
  return (uint8_t *)(header + 1);
}
static uint8_t *path_bytes(const char *path) {
  uint8_t *bytes = bytes_new(strlen(path));
  memcpy(bytes, path, strlen(path)); /* Deliberately no trailing NUL. */
  return bytes;
}
static void bytes_free(uint8_t *bytes) { free(Moonbit_object_header(bytes)); }

static void exchange(lf_tls_session *server, void *client_input,
                     void *client_output, uint8_t *scratch) {
  lf_tls_identity *p = server->identity;
  while (p->bio_ctrl(client_output, 10, 0, NULL) > 0) {
    int n = p->bio_read(client_output, scratch, 32768);
    assert(n > 0 && lunaflux_tls_feed(server, scratch, n) == n);
  }
  for (;;) {
    int n = lunaflux_tls_drain(server, scratch, 32768);
    assert(n >= 0);
    if (n == 0) break;
    assert(p->bio_write(client_input, scratch, n) == n);
  }
}

int main(int argc, char **argv) {
  assert(argc == 3);
  uint8_t *certificate = path_bytes(argv[1]);
  uint8_t *key = path_bytes(argv[2]);
  uint8_t *missing = path_bytes("/no-such-lunaflux-test-key.pem");
  uint8_t *scratch = bytes_new(32768);
  uint8_t *plain = bytes_new(16384);
  uint8_t *received = bytes_new(16384);
  for (int i = 0; i < 16384; i++) plain[i] = (uint8_t)(i % 251);
  for (int iteration = 0; iteration < 16; iteration++) {
    assert(lunaflux_tls_identity_load(certificate, missing) == NULL);
    lf_tls_identity *p = lunaflux_tls_identity_load(certificate, key);
    assert(p != NULL);
    lf_tls_session *abandoned = lunaflux_tls_session_create(p);
    assert(abandoned != NULL && lunaflux_tls_handshake(abandoned) == -2);
    lunaflux_tls_session_release(abandoned);
    assert(p->references == 1);
    lf_tls_session *server = lunaflux_tls_session_create(p);
    assert(server != NULL && p->references == 2);
    const void *(*client_method)(void) = NULL;
    int (*connect_client)(void *) = NULL;
    void *address = dlsym(p->library, "TLS_client_method");
    assert(address != NULL && sizeof(address) == sizeof(client_method));
    memcpy(&client_method, &address, sizeof(address));
    address = dlsym(p->library, "SSL_connect");
    assert(address != NULL && sizeof(address) == sizeof(connect_client));
    memcpy(&connect_client, &address, sizeof(address));
    void *context = p->ctx_new(client_method());
    assert(context != NULL);
    void *client = p->ssl_new(context);
    void *input = p->bio_new(p->bio_method());
    void *output = p->bio_new(p->bio_method());
    assert(client != NULL && input != NULL && output != NULL);
    p->bio_ctrl(input, 130, -1, NULL);
    p->ssl_set_bio(client, input, output);
    /* Explicit identity close must leave the accepted session usable. */
    lunaflux_tls_identity_release(p);
    assert(p->references == 1);
    int server_ready = 0, client_ready = 0;
    for (int step = 0; step < 100 && !(server_ready && client_ready); step++) {
      if (!client_ready) {
        p->clear_error();
        int n = connect_client(client);
        int error = n == 1 ? 0 : p->ssl_error(client, n);
        assert(n == 1 || error == 2 || error == 3);
        client_ready = n == 1;
      }
      exchange(server, input, output, scratch);
      if (!server_ready) {
        int n = lunaflux_tls_handshake(server);
        assert(n == 1 || n == -2 || n == -3);
        server_ready = n == 1;
      }
      exchange(server, input, output, scratch);
    }
    assert(server_ready && client_ready);
    for (int frame = 0; frame < 8; frame++) {
      p->clear_error();
      assert(p->ssl_write(client, plain, 16384) == 16384);
      exchange(server, input, output, scratch);
      assert(lunaflux_tls_read(server, received, 0, 16384) == 16384);
      assert(memcmp(received, plain, 16384) == 0);
      assert(lunaflux_tls_write(server, plain, 0, 16384) == 16384);
      exchange(server, input, output, scratch);
      p->clear_error();
      assert(p->ssl_read(client, received, 16384) == 16384);
      assert(memcmp(received, plain, 16384) == 0);
    }
    p->ssl_free(client);
    p->ctx_free(context);
    lunaflux_tls_session_release(server);
  }
  bytes_free(received); bytes_free(plain); bytes_free(scratch);
  bytes_free(missing); bytes_free(key); bytes_free(certificate);
  puts("16 identity lifecycles, cancelled handshakes and 4 MiB duplex passed");
  return 0;
}
