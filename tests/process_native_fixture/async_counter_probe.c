#define _GNU_SOURCE 1
#include <sys/uio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Read-only diagnostic of two symbol-resolved async counters. No ptrace,
 * no target writes, no payload/model memory, no GPU calls. Addresses are
 * obtained from the exact running executable's ELF symbols and mappings. */
static int read_value(pid_t pid, uintptr_t address, void *out, size_t bytes) {
  struct iovec local = {out, bytes}, remote = {(void *)address, bytes};
  return process_vm_readv(pid, &local, 1, &remote, 1, 0) == (ssize_t)bytes;
}
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  pid_t pid = (pid_t)strtol(argv[1], NULL, 10);
  for (int i = 2; i < 4; ++i) {
    uintptr_t pointer;
    int32_t value;
    uintptr_t address = strtoull(argv[i], NULL, 0);
    if (!read_value(pid,address,&pointer,sizeof(pointer)) ||
        !read_value(pid,pointer,&value,sizeof(value))) return 3;
    printf("%s=%d unsigned=%u\n",i==2?"coroutine_id":"timer_id",value,(uint32_t)value);
  }
  return 0;
}
