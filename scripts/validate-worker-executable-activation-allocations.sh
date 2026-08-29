#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

sources='internal/process/process_approved_spawn.c internal/approved_fs_capability/approved_fs_capability.h'

if rg -n \
  '(^|[^[:alnum:]_])(malloc|calloc|realloc|free|strdup|asprintf|moonbit_make_[[:alnum:]_]+)[[:space:]]*\(' \
  $sources; then
  printf '%s\n' \
    'pinned executable duplicate/fork/exec path must not allocate' >&2
  exit 1
fi

if ! rg -q 'lf_approved_executable_duplicate' \
    internal/approved_fs_capability/approved_fs_capability.h ||
  ! rg -q 'lf_process_spawn_approved' \
    internal/process/process_approved_spawn.c; then
  printf '%s\n' 'pinned executable allocation gate lost its positive scope' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux worker executable activation allocation gate passed.'
