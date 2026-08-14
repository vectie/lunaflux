#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "approved_fs_private.h"

static int32_t lf_reduce_identity_close_status(
  int32_t primary_status,
  int32_t close_status
) {
  if (primary_status != LF_APPROVED_OK) return primary_status;
  return close_status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_require_absolute_identity(
  lf_approved_handle *root,
  moonbit_bytes_t path
) {
  int32_t length = Moonbit_array_length(path);
  if (!lf_validate_path(path, length, 1)) return LF_APPROVED_INVALID;

  int root_fd = -1;
  int32_t status = lf_begin_operation(root, LF_APPROVED_ROOT, &root_fd);
  if (status != LF_APPROVED_OK) return status;

  int starting_fd = open(
    "/",
    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW
  );
  if (starting_fd < 0) {
    status = lf_open_status(errno);
    lf_end_operation(root);
    return status;
  }
  int candidate = lf_traverse(
    starting_fd,
    path,
    1,
    length - 1,
    0,
    &status
  );
  if (candidate < 0) {
    lf_end_operation(root);
    return status;
  }

  struct stat pinned_info;
  struct stat candidate_info;
  if (fstat(root_fd, &pinned_info) != 0 ||
      fstat(candidate, &candidate_info) != 0 ||
      !S_ISDIR(pinned_info.st_mode) ||
      !S_ISDIR(candidate_info.st_mode)) {
    status = LF_APPROVED_FAILED;
  } else if (pinned_info.st_dev != candidate_info.st_dev ||
             pinned_info.st_ino != candidate_info.st_ino) {
    status = LF_APPROVED_IDENTITY_MISMATCH;
  } else {
    status = LF_APPROVED_OK;
  }
  status = lf_reduce_identity_close_status(status, lf_close_fd(candidate));
  lf_end_operation(root);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_approved_fs_test_reduce_identity_close(
  int32_t primary_status,
  int32_t close_status
) {
  return lf_reduce_identity_close_status(primary_status, close_status);
}
