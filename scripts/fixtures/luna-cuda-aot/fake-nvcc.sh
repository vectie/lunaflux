#!/bin/sh
set -eu

if [ "${1:-}" = --version ]; then
  printf 'Cuda compilation tools, release 13.1, V%s\n' "${FAKE_LUNA_CUDA_VERSION:-13.1.0}"
  exit 0
fi
output=
previous=
resource_audit=0
for argument in "$@"; do
  if [ "$previous" = -o ]; then
    output=$argument
  fi
  if [ "$argument" = -Xptxas=-v ]; then
    resource_audit=1
  fi
  previous=$argument
done
[ -n "$output" ] || exit 2
if [ "${FAKE_LUNA_CUDA_NONDETERMINISTIC:-0}" = 1 ]; then
  case "$PWD" in
    *first*) printf '%s\n' 'fake-cubin-first' >"$output" ;;
    *) printf '%s\n' 'fake-cubin-second' >"$output" ;;
  esac
else
  printf '%s\n' 'fake-cubin-deterministic' >"$output"
fi
if [ "$resource_audit" = 1 ]; then
  case "$PWD" in
    *residual*) shared=512 ;;
    *) shared=0 ;;
  esac
  printf '%s\n' \
    "ptxas info    : Used 64 registers, $shared bytes smem" \
    '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' >&2
fi
