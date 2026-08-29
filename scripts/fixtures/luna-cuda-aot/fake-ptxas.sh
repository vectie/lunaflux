#!/bin/sh
set -eu

[ "${1:-}" = --version ] || exit 2
printf 'ptxas release 13.1, V%s\n' "${FAKE_LUNA_CUDA_VERSION:-13.1.0}"
