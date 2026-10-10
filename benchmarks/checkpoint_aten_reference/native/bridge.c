#define _POSIX_C_SOURCE 200809L
#include <moonbit.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Test-only dynamic bridge: neither Torch headers nor its library enter the
 * production MoonBit module. Every backend tensor belongs to this session. */
typedef struct {
  void *library, *context;
  void *(*open)(int, int64_t);
  void (*close)(void *);
  const char *(*error)(void *);
  void (*release)(void *, void *);
  int64_t (*live)(void *);
  void *(*load)(void *, const char *, int64_t, int, int);
  void *(*apply)(void *, int, void *, void *, void *, int, int, int, double);
  int (*write)(void *, void *, const char *);
  char message[1024];
} Bridge;

static void message(Bridge *b, const char *text) {
  snprintf(b->message, sizeof(b->message), "%s", text ? text : "unknown error");
}

MOONBIT_FFI_EXPORT Bridge *lfr_bridge_open(const char *path, int device, int64_t budget) {
  Bridge *b = calloc(1, sizeof(*b));
  if (!b) abort();
  b->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!b->library) { message(b, dlerror()); return b; }
#define RESOLVE(field, symbol) \
  *(void **)(&b->field) = dlsym(b->library, symbol); \
  if (!b->field) { message(b, dlerror()); return b; }
  RESOLVE(open, "lfr_ref_open")
  RESOLVE(close, "lfr_ref_close")
  RESOLVE(error, "lfr_ref_error")
  RESOLVE(release, "lfr_ref_release")
  RESOLVE(live, "lfr_ref_live")
  RESOLVE(load, "lfr_ref_load")
  RESOLVE(apply, "lfr_ref_apply")
  RESOLVE(write, "lfr_ref_write")
#undef RESOLVE
  b->context = b->open(device, budget);
  if (!b->context) message(b, "ATen context initialization failed");
  return b;
}
MOONBIT_FFI_EXPORT int lfr_bridge_valid(Bridge *b) { return b && b->context && !b->message[0]; }
MOONBIT_FFI_EXPORT int lfr_bridge_same(Bridge *a, Bridge *b) { return a == b; }
MOONBIT_FFI_EXPORT void lfr_bridge_close(Bridge *b) {
  if (!b) return;
  if (b->context) b->close(b->context);
  if (b->library) dlclose(b->library);
  free(b);
}
MOONBIT_FFI_EXPORT moonbit_bytes_t lfr_bridge_error(Bridge *b) {
  const char *s = b->message;
  int32_t n = (int32_t)strlen(s);
  moonbit_bytes_t bytes = moonbit_make_bytes(n, 0);
  memcpy(bytes, s, (size_t)n);
  return bytes;
}
MOONBIT_FFI_EXPORT void *lfr_bridge_null(void) { return NULL; }
MOONBIT_FFI_EXPORT int lfr_bridge_tensor_valid(void *tensor) { return tensor != NULL; }
MOONBIT_FFI_EXPORT void lfr_bridge_tensor_close(Bridge *b, void *t) { b->release(b->context, t); }
MOONBIT_FFI_EXPORT int64_t lfr_bridge_live_bytes(Bridge *b) { return b->live(b->context); }
MOONBIT_FFI_EXPORT void *lfr_bridge_load(Bridge *b, const char *p, int64_t offset, int rows, int cols) {
  b->message[0] = 0;
  void *tensor = b->load(b->context, p, offset, rows, cols);
  if (!tensor) message(b, b->error(b->context));
  return tensor;
}
MOONBIT_FFI_EXPORT void *lfr_bridge_apply(Bridge *b, int op, void *x, void *y, void *z, int a, int c, int d, double e) {
  b->message[0] = 0;
  void *tensor = b->apply(b->context, op, x, y, z, a, c, d, e);
  if (!tensor) message(b, b->error(b->context));
  return tensor;
}
MOONBIT_FFI_EXPORT int lfr_bridge_write(Bridge *b, void *t, const char *p) {
  b->message[0] = 0;
  int status = b->write(b->context, t, p);
  if (status != 0) message(b, b->error(b->context));
  return status;
}
MOONBIT_FFI_EXPORT moonbit_bytes_t lfr_bridge_read(Bridge *b, const char *path, int64_t offset, int length) {
  b->message[0] = 0;
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0) { message(b, strerror(errno)); return moonbit_make_bytes(0, 0); }
  if (fstat(fd, &st) || offset < 0 || offset > st.st_size) {
    message(b, "invalid file extent"); close(fd); return moonbit_make_bytes(0, 0);
  }
  int64_t count = length < 0 ? st.st_size - offset : length;
  if (count < 0 || count > 2097152 || count > st.st_size - offset) {
    message(b, "metadata read exceeds extent/budget"); close(fd); return moonbit_make_bytes(0, 0);
  }
  moonbit_bytes_t out = moonbit_make_bytes((int32_t)count, 0);
  int64_t done = 0;
  while (done < count) {
    ssize_t n = pread(fd, out + done, (size_t)(count - done), offset + done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { message(b, "incomplete metadata read"); break; }
    done += n;
  }
  close(fd);
  return out;
}
