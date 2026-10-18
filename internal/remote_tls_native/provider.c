#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <moonbit.h>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>

/* OpenSSL's stable C ABI. The provider uses TLS 1.3 and memory BIOs; socket
 * readiness, cancellation, and deadlines remain in the MoonBit async owner. */
typedef struct {
  void *library;
  void *context;
  size_t references;
  const void *(*server_method)(void);
  void *(*ctx_new)(const void *);
  void (*ctx_free)(void *);
  int (*ctx_certificate)(void *, const char *);
  int (*ctx_key)(void *, const char *, int);
  int (*ctx_check_key)(const void *);
  long (*ctx_ctrl)(void *, int, long, void *);
  void *(*ssl_new)(void *);
  void (*ssl_free)(void *);
  void (*ssl_set_bio)(void *, void *, void *);
  int (*ssl_accept)(void *);
  int (*ssl_read)(void *, void *, int);
  int (*ssl_write)(void *, const void *, int);
  int (*ssl_error)(const void *, int);
  void (*clear_error)(void);
  const void *(*bio_method)(void);
  void *(*bio_new)(const void *);
  int (*bio_free)(void *);
  int (*bio_read)(void *, void *, int);
  int (*bio_write)(void *, const void *, int);
  long (*bio_ctrl)(void *, int, long, void *);
} lf_tls_identity;

typedef struct {
  lf_tls_identity *identity;
  void *ssl;
  void *input;
  void *output;
} lf_tls_session;

static void *lf_tls_library(void) {
#if defined(__APPLE__)
  const char *names[] = {
    "/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib",
    "/usr/local/opt/openssl@3/lib/libssl.3.dylib",
    "libssl.3.dylib", "libssl.dylib"
  };
#else
  const char *names[] = { "libssl.so.3", "libssl.so.1.1" };
#endif
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    void *handle = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (handle != NULL) return handle;
  }
  return NULL;
}

static void lf_tls_identity_release(lf_tls_identity *p) {
  if (p == NULL) return;
  if (--p->references != 0) return;
  if (p->context != NULL) p->ctx_free(p->context);
  if (p->library != NULL) dlclose(p->library);
  memset(p, 0, sizeof(*p));
  free(p);
}

#define LF_TLS_SYMBOL(field, symbol) do { \
  void *address = dlsym(p->library, symbol); \
  if (address == NULL || sizeof(p->field) != sizeof(address)) goto fail; \
  memcpy(&p->field, &address, sizeof(address)); \
} while (0)

/* MoonBit byte arrays have an explicit length. OpenSSL's file API instead
 * requires a terminated C string; materialize it only during startup. */
static char *lf_tls_path(const uint8_t *bytes) {
  size_t length = (size_t)Moonbit_array_length(bytes);
  char *path = malloc(length + 1);
  if (path == NULL) return NULL;
  memcpy(path, bytes, length);
  path[length] = '\0';
  return path;
}

MOONBIT_FFI_EXPORT void *lunaflux_tls_identity_load(
    const uint8_t *certificate, const uint8_t *key) {
  if (certificate == NULL || key == NULL ||
      Moonbit_array_length(certificate) == 0 || Moonbit_array_length(key) == 0 ||
      memchr(certificate, 0, Moonbit_array_length(certificate)) != NULL ||
      memchr(key, 0, Moonbit_array_length(key)) != NULL) return NULL;
  lf_tls_identity *p = calloc(1, sizeof(*p));
  if (p == NULL) return NULL;
  char *certificate_path = lf_tls_path(certificate);
  char *key_path = lf_tls_path(key);
  p->references = 1;
  if (certificate_path == NULL || key_path == NULL) goto fail;
  p->library = lf_tls_library();
  if (p->library == NULL) goto fail;
  LF_TLS_SYMBOL(server_method, "TLS_server_method");
  LF_TLS_SYMBOL(ctx_new, "SSL_CTX_new");
  LF_TLS_SYMBOL(ctx_free, "SSL_CTX_free");
  LF_TLS_SYMBOL(ctx_certificate, "SSL_CTX_use_certificate_chain_file");
  LF_TLS_SYMBOL(ctx_key, "SSL_CTX_use_PrivateKey_file");
  LF_TLS_SYMBOL(ctx_check_key, "SSL_CTX_check_private_key");
  LF_TLS_SYMBOL(ctx_ctrl, "SSL_CTX_ctrl");
  LF_TLS_SYMBOL(ssl_new, "SSL_new");
  LF_TLS_SYMBOL(ssl_free, "SSL_free");
  LF_TLS_SYMBOL(ssl_set_bio, "SSL_set_bio");
  LF_TLS_SYMBOL(ssl_accept, "SSL_accept");
  LF_TLS_SYMBOL(ssl_read, "SSL_read");
  LF_TLS_SYMBOL(ssl_write, "SSL_write");
  LF_TLS_SYMBOL(ssl_error, "SSL_get_error");
  LF_TLS_SYMBOL(clear_error, "ERR_clear_error");
  LF_TLS_SYMBOL(bio_method, "BIO_s_mem");
  LF_TLS_SYMBOL(bio_new, "BIO_new");
  LF_TLS_SYMBOL(bio_free, "BIO_free");
  LF_TLS_SYMBOL(bio_read, "BIO_read");
  LF_TLS_SYMBOL(bio_write, "BIO_write");
  LF_TLS_SYMBOL(bio_ctrl, "BIO_ctrl");
  p->context = p->ctx_new(p->server_method());
  if (p->context == NULL) goto fail;
  /* SSL_CTRL_SET_MIN/MAX_PROTO_VERSION; TLS1_3_VERSION. */
  if (p->ctx_ctrl(p->context, 123, 0x0304, NULL) != 1 ||
      p->ctx_ctrl(p->context, 124, 0x0304, NULL) != 1 ||
      p->ctx_certificate(p->context, certificate_path) != 1 ||
      p->ctx_key(p->context, key_path, 1) != 1 ||
      p->ctx_check_key(p->context) != 1) goto fail;
  free(certificate_path);
  free(key_path);
  return p;
fail:
  free(certificate_path);
  free(key_path);
  lf_tls_identity_release(p);
  return NULL;
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_identity_valid(void *p) {
  return p != NULL;
}

MOONBIT_FFI_EXPORT void lunaflux_tls_identity_release(void *p) {
  lf_tls_identity_release(p);
}

MOONBIT_FFI_EXPORT void *lunaflux_tls_session_create(void *identity) {
  lf_tls_identity *p = identity;
  if (p == NULL || p->references == SIZE_MAX) return NULL;
  lf_tls_session *s = calloc(1, sizeof(*s));
  if (s == NULL) return NULL;
  s->identity = p;
  s->ssl = p->ssl_new(p->context);
  s->input = p->bio_new(p->bio_method());
  s->output = p->bio_new(p->bio_method());
  if (s->ssl == NULL || s->input == NULL || s->output == NULL) {
    if (s->ssl != NULL) p->ssl_free(s->ssl);
    if (s->input != NULL) p->bio_free(s->input);
    if (s->output != NULL) p->bio_free(s->output);
    free(s);
    return NULL;
  }
  /* BIO_C_SET_BUF_MEM_EOF_RETURN: empty input means retry, never EOF. */
  p->bio_ctrl(s->input, 130, -1, NULL);
  p->ssl_set_bio(s->ssl, s->input, s->output);
  p->references++;
  return s;
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_session_valid(void *p) {
  return p != NULL;
}

MOONBIT_FFI_EXPORT void lunaflux_tls_session_release(void *session) {
  lf_tls_session *s = session;
  if (s == NULL) return;
  lf_tls_identity *p = s->identity;
  p->ssl_free(s->ssl); /* SSL owns both BIOs. */
  memset(s, 0, sizeof(*s));
  free(s);
  lf_tls_identity_release(p);
}

static int lf_tls_result(lf_tls_session *s, int result) {
  if (result > 0) return result;
  /* SSL_get_error must immediately follow the operation on this SSL. */
  int error = s->identity->ssl_error(s->ssl, result);
  if (error == 2) return -2; /* WANT_READ */
  if (error == 3) return -3; /* WANT_WRITE */
  if (error == 6) return 0;  /* ZERO_RETURN */
  return -1;
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_handshake(void *session) {
  lf_tls_session *s = session;
  if (s == NULL) return -1;
  s->identity->clear_error();
  return lf_tls_result(s, s->identity->ssl_accept(s->ssl));
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_feed(
    void *session, const uint8_t *bytes, int32_t count) {
  lf_tls_session *s = session;
  if (s == NULL || count <= 0 || count > Moonbit_array_length(bytes)) return -1;
  return s->identity->bio_write(s->input, bytes, count);
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_drain(
    void *session, uint8_t *bytes, int32_t capacity) {
  lf_tls_session *s = session;
  if (s == NULL || capacity <= 0 || capacity > Moonbit_array_length(bytes)) return -1;
  /* BIO_CTRL_PENDING. BIO_read's empty retry must not become a transport error. */
  if (s->identity->bio_ctrl(s->output, 10, 0, NULL) == 0) return 0;
  return s->identity->bio_read(s->output, bytes, capacity);
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_read(
    void *session, uint8_t *bytes, int32_t offset, int32_t count) {
  lf_tls_session *s = session;
  if (s == NULL || offset < 0 || count <= 0 ||
      offset > Moonbit_array_length(bytes) - count) return -1;
  s->identity->clear_error();
  return lf_tls_result(s, s->identity->ssl_read(s->ssl, bytes + offset, count));
}

MOONBIT_FFI_EXPORT int32_t lunaflux_tls_write(
    void *session, const uint8_t *bytes, int32_t offset, int32_t count) {
  lf_tls_session *s = session;
  if (s == NULL || offset < 0 || count <= 0 ||
      offset > Moonbit_array_length(bytes) - count) return -1;
  s->identity->clear_error();
  return lf_tls_result(s, s->identity->ssl_write(s->ssl, bytes + offset, count));
}
#else
MOONBIT_FFI_EXPORT void *lunaflux_tls_identity_load(const uint8_t *c, const uint8_t *k) { (void)c; (void)k; return NULL; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_identity_valid(void *p) { (void)p; return 0; }
MOONBIT_FFI_EXPORT void lunaflux_tls_identity_release(void *p) { (void)p; }
MOONBIT_FFI_EXPORT void *lunaflux_tls_session_create(void *p) { (void)p; return NULL; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_session_valid(void *p) { (void)p; return 0; }
MOONBIT_FFI_EXPORT void lunaflux_tls_session_release(void *p) { (void)p; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_handshake(void *p) { (void)p; return -1; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_feed(void *p, const uint8_t *b, int32_t n) { (void)p; (void)b; (void)n; return -1; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_drain(void *p, uint8_t *b, int32_t n) { (void)p; (void)b; (void)n; return -1; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_read(void *p, uint8_t *b, int32_t o, int32_t n) { (void)p; (void)b; (void)o; (void)n; return -1; }
MOONBIT_FFI_EXPORT int32_t lunaflux_tls_write(void *p, const uint8_t *b, int32_t o, int32_t n) { (void)p; (void)b; (void)o; (void)n; return -1; }
#endif
