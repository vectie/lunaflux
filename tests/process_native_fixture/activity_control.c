#define _GNU_SOURCE 1
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/un.h>

/* One bounded local datagram transaction for the offline collector. All
 * campaign orchestration stays in MoonBit; this fixture only bridges AF_UNIX. */
int main(int argc, char **argv) {
  if (argc != 3 || (strcmp(argv[2], "start") && strcmp(argv[2], "stop") &&
                   strcmp(argv[2], "status") && strcmp(argv[2], "seal"))) return 2;
  struct sockaddr_un local = {.sun_family = AF_UNIX};
  struct sockaddr_un remote = {.sun_family = AF_UNIX};
  if (strlen(argv[1]) >= sizeof(remote.sun_path)) return 2;
  strcpy(remote.sun_path, argv[1]);
  // Linux abstract endpoint: host-network reference containers can reply
  // without sharing the controller's /tmp mount namespace.
  snprintf(local.sun_path + 1, sizeof(local.sun_path) - 1, "lf-activity-client-%d", getpid());
  socklen_t local_size = (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
                                    1 + strlen(local.sun_path + 1));
  int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return 3;
  struct timeval timeout = {.tv_sec = 15};
  int result = 3;
  if (bind(fd, (struct sockaddr *)&local, local_size) ||
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout))) goto done;
  if (sendto(fd, argv[2], strlen(argv[2]), 0,
             (struct sockaddr *)&remote, sizeof(remote)) < 0) goto done;
  char reply[2048];
  ssize_t n = recv(fd, reply, sizeof(reply), 0);
  if (n > 0 && fwrite(reply, 1, (size_t)n, stdout) == (size_t)n) result = 0;
done:
  close(fd); return result;
}
