#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

if ! rg -q 'openat\s*\(' internal/approved_fs/approved_fs.c ||
  ! rg -q 'O_NOFOLLOW' internal/approved_fs/approved_fs.c ||
  ! rg -q 'O_DIRECTORY' internal/approved_fs/approved_fs.c ||
  ! rg -q 'O_NONBLOCK' internal/approved_fs/approved_fs.c ||
  ! rg -q 'fstat\s*\(' internal/approved_fs/approved_fs.c ||
  ! rg -q 'pread\s*\(' internal/approved_fs/approved_fs.c ||
  ! rg -q 'lunaflux_approved_fs_read_immutable_snapshot' \
    internal/approved_fs/approved_fs.c ||
  ! rg -q 'lf_file_stamp_equal' internal/approved_fs/approved_fs.c ||
  ! rg -q 'stdatomic\.h' internal/approved_fs/approved_fs.c ||
  ! rg -q 'lf_begin_operation' internal/approved_fs/approved_fs.c ||
  ! rg -q '_Static_assert\(sizeof\(off_t\)' \
    internal/approved_fs/approved_fs_private.h; then
  printf '%s\n' \
    'approved filesystem ABI is missing no-follow traversal, type checks, or positional reads' >&2
  failed=1
fi

if rg -n 'supported_targets|supported-targets' runtime/approved_fs/moon.pkg; then
  printf '%s\n' 'approved filesystem must gate native files, not the package' >&2
  failed=1
fi

if matches=$(rg -n 'extern\s+"[cC]"' runtime/approved_fs --glob '*.mbt' 2>/dev/null); then
  printf '%s\n%s\n' \
    'approved filesystem extern declarations escaped the internal ABI package:' \
    "$matches" >&2
  failed=1
fi

if ! rg -q '#borrow\(path, status\)' internal/approved_fs/ffi.mbt ||
  ! rg -q '#borrow\(root, locator, status\)' internal/approved_fs/ffi.mbt ||
  ! rg -q '#borrow\(file, destination\)' internal/approved_fs/ffi.mbt ||
  ! rg -q '#borrow\(file, output\)' internal/approved_fs/ffi.mbt ||
  ! rg -q '#borrow\(file, failure, status\)' internal/approved_fs/ffi.mbt; then
  printf '%s\n' 'approved filesystem FFI ownership annotations are incomplete' >&2
  failed=1
fi

snapshot_allocations=$(rg -c 'moonbit_make_bytes\s*\(' \
  internal/approved_fs/approved_fs.c || true)
if [ "$snapshot_allocations" -ne 1 ]; then
  printf '%s\n' \
    'approved immutable snapshot must have exactly one payload-allocation site' >&2
  failed=1
fi

for type_name in ApprovedRoot ApprovedFile; do
  interface_block=$(sed -n "/^pub struct $type_name {/,/^$/p" \
    runtime/approved_fs/pkg.generated.mbti)
  if [ -z "$interface_block" ] || \
    printf '%s\n' "$interface_block" | rg -q 'Debug|descriptor|fd|path|handle'; then
    printf '%s: %s\n' \
      'approved filesystem owner leaks Debug or native/path authority' \
      "$type_name" >&2
    failed=1
  fi
done

for source_file in runtime/approved_fs/api.mbt runtime/approved_fs/path.mbt \
  internal/approved_fs/approved_fs.c internal/approved_fs/api.mbt \
  internal/approved_fs/asan_probe.c \
  internal/approved_fs/approved_fs_private.h; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; approved filesystem boundary files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux approved filesystem ABI boundary is valid.'
