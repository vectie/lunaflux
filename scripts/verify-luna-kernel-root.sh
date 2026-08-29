#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/luna-bf16-kernel-producer-common.sh"

[ "$#" -eq 1 ] || lbf_fail 'usage: verify-luna-kernel-root.sh ABSOLUTE_ASSEMBLED_ROOT'
root=$1
lbf_require_absolute_directory "$root"
top=$(find "$root" -mindepth 1 -maxdepth 1 -print | sed "s#^$root/##" | LC_ALL=C sort)
[ "$top" = "kernel-root
kernel-root.plan.v1
kernel.files.sha256" ] || lbf_fail 'assembled kernel root has unexpected entries'
if find "$root" -type l -print | grep -q .; then
  lbf_fail 'assembled kernel root contains a symbolic link'
fi
if find "$root" ! -type d ! -type f -print | grep -q .; then
  lbf_fail 'assembled kernel root contains a special object'
fi
plan_sha=$(lbf_sha256_file "$root/kernel-root.plan.v1")
scratch_parent=$(realpath -- "$(mktemp -d /tmp/lunaflux-kernel-root-reverify.XXXXXX)") ||
  lbf_fail 'could not create re-verification scratch'
cleanup() {
  chmod -R u+w "$scratch_parent" 2>/dev/null || true
  rm -rf -- "$scratch_parent"
}
trap cleanup EXIT HUP INT TERM
source=$scratch_parent/source
mkdir "$source"
cp "$root/kernel-root.plan.v1" "$source/kernel-root.plan.v1"
cp "$root/kernel.files.sha256" "$source/kernel.files.sha256"
mkdir "$source/payload"
cp -R "$root/kernel-root/." "$source/payload/"
# Reuse the authoritative assembler validation without permitting publication
# over caller data. A fresh scratch output proves the complete source contract.
LUNA_KERNEL_ROOT_VERIFY_INNER=1 "$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$source#sha256=$plan_sha" "$scratch_parent/verified" >/dev/null
chmod -R u+w "$scratch_parent/verified" 2>/dev/null || true
rm -rf -- "$scratch_parent/verified"
printf '%s\n' 'Luna deployment kernel-root verification passed'
