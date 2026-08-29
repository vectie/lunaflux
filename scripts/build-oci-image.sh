#!/bin/sh

set -eu

usage() {
  printf '%s\n' \
    'usage: build-oci-image.sh BASE_IMAGE CONTEXT_DIR PLATFORM OUTPUT_OCI_TAR' >&2
  exit 2
}

[ "$#" -eq 4 ] || usage
base_image=$1
context_dir=$2
platform=$3
output=$4
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

[ "$(uname -s)" = Linux ] || {
  printf '%s\n' 'OCI release construction requires an approved Linux builder.' >&2
  exit 1
}
case "$platform" in
  linux/amd64) expected_architecture=x86_64 ;;
  linux/arm64) expected_architecture=aarch64 ;;
  *) printf '%s\n' 'platform must be linux/amd64 or linux/arm64' >&2; exit 1 ;;
esac
case "$output" in
  /*.tar) ;;
  *) printf '%s\n' 'output must be a new absolute .tar path' >&2; exit 1 ;;
esac
[ ! -e "$output" ] || {
  printf '%s\n' 'refusing to overwrite an existing OCI archive' >&2
  exit 1
}

# This call is the authority-bearing precondition for the build below. It
# compares BASE_IMAGE with metadata/base-image.ref and verifies the exact
# staged payload. The Containerfile cannot perform this host-side check.
"$repo_root/scripts/verify-oci-context.sh" "$base_image" "$context_dir"
architecture=$(sed -n '1p' "$context_dir/metadata/linux-architecture")
[ "$architecture" = "$expected_architecture" ] || {
  printf '%s\n' 'requested platform does not match verified context metadata' >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || {
  printf '%s\n' 'docker buildx is required on the approved Linux builder' >&2
  exit 1
}
docker buildx version >/dev/null
docker buildx build --pull --platform "$platform" \
  --build-arg "BASE_IMAGE=$base_image" \
  --file "$repo_root/deploy/oci/Containerfile" \
  --output "type=oci,dest=$output" "$context_dir"
