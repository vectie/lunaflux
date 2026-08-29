#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

typedef struct {
  uint32_t state[8];
  uint64_t bytes;
  unsigned char block[64];
  size_t used;
} Sha256;

static const uint32_t k[64] = {
  0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
  0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
  0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
  0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
  0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
  0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
  0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
  0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
  0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
  0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
  0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
  0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
  0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
  0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
  0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
  0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

static uint32_t rotr(uint32_t value, unsigned count) {
  return (value >> count) | (value << (32U - count));
}

static void sha256_transform(Sha256 *ctx, const unsigned char block[64]) {
  uint32_t w[64];
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0; i < 16; ++i) {
    size_t offset = i * 4;
    w[i] = ((uint32_t)block[offset] << 24) |
      ((uint32_t)block[offset + 1] << 16) |
      ((uint32_t)block[offset + 2] << 8) | block[offset + 3];
  }
  for (i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
      (w[i - 15] >> 3);
    uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
      (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
  e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
  for (i = 0; i < 64; ++i) {
    uint32_t sum1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    uint32_t choose = (e & f) ^ ((~e) & g);
    uint32_t t1 = h + sum1 + choose + k[i] + w[i];
    uint32_t sum0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = sum0 + majority;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c;
  ctx->state[3] += d; ctx->state[4] += e; ctx->state[5] += f;
  ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256 *ctx) {
  static const uint32_t initial[8] = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  memcpy(ctx->state, initial, sizeof(initial));
  ctx->bytes = 0;
  ctx->used = 0;
}

static void sha256_update(Sha256 *ctx, const unsigned char *data, size_t size) {
  ctx->bytes += size;
  while (size > 0) {
    size_t available = 64 - ctx->used;
    size_t take = size < available ? size : available;
    memcpy(ctx->block + ctx->used, data, take);
    ctx->used += take;
    data += take;
    size -= take;
    if (ctx->used == 64) {
      sha256_transform(ctx, ctx->block);
      ctx->used = 0;
    }
  }
}

static void sha256_final(Sha256 *ctx, unsigned char digest[32]) {
  uint64_t bits = ctx->bytes * 8;
  size_t i;
  ctx->block[ctx->used++] = 0x80;
  if (ctx->used > 56) {
    memset(ctx->block + ctx->used, 0, 64 - ctx->used);
    sha256_transform(ctx, ctx->block);
    ctx->used = 0;
  }
  memset(ctx->block + ctx->used, 0, 56 - ctx->used);
  for (i = 0; i < 8; ++i) {
    ctx->block[63 - i] = (unsigned char)(bits >> (i * 8));
  }
  sha256_transform(ctx, ctx->block);
  for (i = 0; i < 8; ++i) {
    digest[i * 4] = (unsigned char)(ctx->state[i] >> 24);
    digest[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
    digest[i * 4 + 2] = (unsigned char)(ctx->state[i] >> 8);
    digest[i * 4 + 3] = (unsigned char)ctx->state[i];
  }
}

static int decode_digest(const char *text, unsigned char digest[32]) {
  size_t i;
  if (strlen(text) != 64) return -1;
  for (i = 0; i < 32; ++i) {
    int high = text[i * 2];
    int low = text[i * 2 + 1];
    high = high >= '0' && high <= '9' ? high - '0' :
      high >= 'a' && high <= 'f' ? high - 'a' + 10 : -1;
    low = low >= '0' && low <= '9' ? low - '0' :
      low >= 'a' && low <= 'f' ? low - 'a' + 10 : -1;
    if (high < 0 || low < 0) return -1;
    digest[i] = (unsigned char)((high << 4) | low);
  }
  return 0;
}

static int hash_fd(int fd, unsigned char digest[32]) {
  unsigned char buffer[16384];
  struct stat status;
  Sha256 ctx;
  off_t offset = 0;
  if (fstat(fd, &status) < 0 || !S_ISREG(status.st_mode) ||
      status.st_nlink > 1 || status.st_size <= 0) return -1;
  sha256_init(&ctx);
  for (;;) {
    ssize_t count = pread(fd, buffer, sizeof(buffer), offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return -1;
    if (count == 0) break;
    sha256_update(&ctx, buffer, (size_t)count);
    offset += count;
  }
  sha256_final(&ctx, digest);
  return 0;
}

static int digest_matches(int fd, const char *expected) {
  unsigned char actual[32], decoded[32];
  unsigned difference = 0;
  size_t i;
  if (decode_digest(expected, decoded) < 0 || hash_fd(fd, actual) < 0) return 0;
  for (i = 0; i < 32; ++i) difference |= actual[i] ^ decoded[i];
  return difference == 0;
}

static int open_pinned(const char *path, const char *digest) {
  struct stat status;
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0 || fstat(fd, &status) < 0 || status.st_nlink != 1 ||
      !digest_matches(fd, digest)) {
    if (fd >= 0) close(fd);
    return -1;
  }
  return fd;
}

#if defined(__linux__) || defined(PHASE4_TEST_SHELL_GATE)
static int copy_bytes(int source, int target) {
  unsigned char buffer[16384];
  off_t offset = 0;
  for (;;) {
    ssize_t count = pread(source, buffer, sizeof(buffer), offset);
    ssize_t written = 0;
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) return -1;
    if (count == 0) return 0;
    while (written < count) {
      ssize_t result = write(target, buffer + written, (size_t)(count - written));
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) return -1;
      written += result;
    }
    offset += count;
  }
}

static int sealed_copy(int source, int executable) {
  int target;
#if defined(__linux__)
  int expected_seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
  target = (int)syscall(SYS_memfd_create, "lunaflux-phase4-pinned",
                        MFD_CLOEXEC | MFD_ALLOW_SEALING);
#elif defined(PHASE4_TEST_SHELL_GATE)
  char path[] = "/tmp/lunaflux-phase4-test-pinned.XXXXXX";
  target = mkstemp(path);
  if (target >= 0) {
    unlink(path);
    fcntl(target, F_SETFD, FD_CLOEXEC);
  }
#else
  (void)source;
  (void)executable;
  return -1;
#endif
  if (target < 0 || copy_bytes(source, target) < 0 ||
      fchmod(target, executable ? 0500 : 0400) < 0) {
    if (target >= 0) close(target);
    return -1;
  }
#if defined(__linux__)
  if (fcntl(target, F_ADD_SEALS, expected_seals) < 0 ||
      fcntl(target, F_GET_SEALS) != expected_seals) {
    close(target);
    return -1;
  }
#endif
  if (lseek(target, 0, SEEK_SET) != 0) {
    close(target);
    return -1;
  }
  return target;
}
#endif

static int exchange_barrier(const char *ready_path, const char *go_path) {
  int ready_fd, go_fd, flags;
  char signal;
  go_fd = open(go_path, O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
  if (go_fd < 0) return -1;
  flags = fcntl(go_fd, F_GETFL);
  if (flags < 0 || fcntl(go_fd, F_SETFL, flags & ~O_NONBLOCK) < 0) return -1;
  ready_fd = open(ready_path, O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
  if (ready_fd < 0 || write(ready_fd, "R\n", 2) != 2) return -1;
  close(ready_fd);
  if (read(go_fd, &signal, 1) != 1 || signal != 'G') return -1;
  close(go_fd);
  return 0;
}

static void close_unowned_descriptors(void) {
#if defined(__linux__) && defined(SYS_close_range)
  if (syscall(SYS_close_range, 3U, 7U, 0U) == 0 &&
      syscall(SYS_close_range, 10U, UINT_MAX, 0U) == 0) return;
#endif
  {
    struct rlimit limit;
    rlim_t fd, maximum;
    long open_max;
    if (getrlimit(RLIMIT_NOFILE, &limit) < 0) _exit(72);
    maximum = limit.rlim_cur;
    if (maximum == RLIM_INFINITY || maximum > INT_MAX) {
      open_max = sysconf(_SC_OPEN_MAX);
      if (open_max < 0) _exit(72);
      maximum = (rlim_t)open_max;
    }
    for (fd = 3; fd < maximum; ++fd) {
      if (fd != 8 && fd != 9) close((int)fd);
    }
  }
}

int main(int argc, char **argv) {
  int gate_source, policy_source, gate_fd, policy_fd;
#if defined(__linux__) || defined(PHASE4_TEST_SHELL_GATE)
  char *clean_environment[] = { "LC_ALL=C", NULL };
#endif
  if (argc < 9) return 64;
  gate_source = open_pinned(argv[1], argv[2]);
  policy_source = open_pinned(argv[3], argv[4]);
  if (gate_source < 0 || policy_source < 0) return 65;
#if defined(__linux__) || defined(PHASE4_TEST_SHELL_GATE)
  gate_fd = sealed_copy(gate_source, 1);
  policy_fd = sealed_copy(policy_source, 0);
  close(gate_source);
  close(policy_source);
  if (gate_fd < 0 || policy_fd < 0 ||
      !digest_matches(gate_fd, argv[2]) ||
      !digest_matches(policy_fd, argv[4])) return 65;
#else
  gate_fd = gate_source;
  policy_fd = policy_source;
#endif
  if (exchange_barrier(argv[5], argv[6]) < 0) return 66;
  if (!digest_matches(gate_fd, argv[2]) ||
      !digest_matches(policy_fd, argv[4])) return 67;
  if (dup2(policy_fd, 8) < 0 || dup2(gate_fd, 9) < 0) return 68;
  close_unowned_descriptors();
#if defined(PHASE4_TEST_SHELL_GATE)
  {
    int i;
    char **shell_argv = calloc((size_t)argc - 5, sizeof(char *));
    if (shell_argv == NULL) return 70;
    shell_argv[0] = "/bin/sh";
    shell_argv[1] = "/dev/fd/9";
    for (i = 8; i < argc; ++i) shell_argv[i - 6] = argv[i];
    execve("/bin/sh", shell_argv, clean_environment);
  }
#elif defined(__linux__)
  syscall(SYS_execveat, 9, "", &argv[7], clean_environment, AT_EMPTY_PATH);
#else
  fputs("Phase 4 FD execution is supported only on Linux\n", stderr);
#endif
  return 71;
}
