#!/bin/sh
set -eu

fail() {
  printf 'luna CUDA AOT driver inspection failed: %s\n' "$1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail 'sha256sum or shasum is required'
  fi
}

[ "$#" -eq 1 ] || fail 'usage: inspect-luna-cuda-aot-driver.sh ABSOLUTE_NVCC'
nvcc=$1
case "$nvcc" in /*) ;; *) fail 'NVCC path must be absolute' ;; esac
[ "$(realpath -- "$nvcc")" = "$nvcc" ] || fail 'NVCC path must be canonical'
[ -f "$nvcc" ] && [ -x "$nvcc" ] && [ ! -L "$nvcc" ] ||
  fail 'NVCC must be an executable non-symlink file'
ptxas=$(dirname -- "$nvcc")/ptxas
[ "$(realpath -- "$ptxas")" = "$ptxas" ] || fail 'ptxas path must be canonical'
[ -f "$ptxas" ] && [ -x "$ptxas" ] && [ ! -L "$ptxas" ] ||
  fail 'sibling ptxas must be an executable non-symlink file'

identity_tmp=$(mktemp -d /tmp/lunaflux-luna-cuda-driver.XXXXXX)
trap 'rm -rf -- "$identity_tmp"' EXIT HUP INT TERM
LC_ALL=C "$nvcc" --version >"$identity_tmp/nvcc-version.txt" 2>&1 ||
  fail 'NVCC version inspection failed'
LC_ALL=C "$ptxas" --version >"$identity_tmp/ptxas-version.txt" 2>&1 ||
  fail 'ptxas version inspection failed'
compiler_version=$(sed -n \
  's/.*V\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
  "$identity_tmp/nvcc-version.txt" | head -n 1)
[ -n "$compiler_version" ] || fail 'NVCC full version is not recognizable'

{
  printf '%s\n' 'schema=lunaflux-luna-cuda-driver-identity-v1'
  printf 'nvcc_sha256=%s\n' "$(sha256_file "$nvcc")"
  printf 'ptxas_sha256=%s\n' "$(sha256_file "$ptxas")"
  printf 'nvcc_version_sha256=%s\n' \
    "$(sha256_file "$identity_tmp/nvcc-version.txt")"
  printf 'ptxas_version_sha256=%s\n' \
    "$(sha256_file "$identity_tmp/ptxas-version.txt")"
} >"$identity_tmp/identity.txt"
cat "$identity_tmp/identity.txt"
printf 'driver_identity_sha256=%s\n' \
  "$(sha256_file "$identity_tmp/identity.txt")"
printf 'compiler_version=%s\n' "$compiler_version"
