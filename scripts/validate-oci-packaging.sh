#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
containerfile=$repo_root/deploy/oci/Containerfile
verifier=$repo_root/scripts/verify-oci-context.sh
builder=$repo_root/scripts/build-oci-image.sh
license_assembler=$repo_root/scripts/assemble-oci-license-inventory.sh

fail() {
  printf '%s\n' "LunaFlux OCI packaging gate failed: $1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail 'no SHA-256 utility is available'
  fi
}

assert_line() {
  grep -F -x "$1" "$containerfile" >/dev/null 2>&1 ||
    fail "Containerfile is missing exact line: $1"
}

assert_rejected() {
  label=$1
  base=$2
  context=$3
  if "$verifier" "$base" "$context" >/dev/null 2>&1; then
    fail "host verifier passed hostile fixture: $label"
  fi
}

write_inventory() {
  context=$1
  inventory=$context/metadata/artifacts.sha256
  chmod 644 "$inventory" 2>/dev/null || true
  : > "$inventory"
  find "$context/rootfs" -type f -print |
    sed "s#^$context/rootfs/##" | LC_ALL=C sort |
    while IFS= read -r relative; do
      printf '%s  %s\n' "$(sha256_file "$context/rootfs/$relative")" \
        "$relative" >> "$inventory"
    done
  chmod 444 "$inventory"
}

sh -n "$verifier"
sh -n "$builder"
sh -n "$license_assembler"
[ "$(grep -c 'verify-oci-context.sh.*base_image.*context_dir' "$builder")" -eq 1 ] ||
  fail 'Linux build wrapper must invoke the host verifier exactly once'
verifier_line=$(grep -n 'verify-oci-context.sh.*base_image.*context_dir' "$builder" |
  cut -d: -f1)
build_line=$(grep -n '^docker buildx build ' "$builder" | cut -d: -f1)
[ -n "$build_line" ] && [ "$verifier_line" -lt "$build_line" ] ||
  fail 'Linux build wrapper must verify before invoking buildx'
[ "$(grep -c '^ARG BASE_IMAGE$' "$containerfile")" -eq 1 ] ||
  fail 'Containerfile must declare exactly one mandatory BASE_IMAGE argument'
[ "$(grep -c '^ARG ' "$containerfile")" -eq 1 ] ||
  fail 'Containerfile must not accept another build argument'
[ "$(grep -c '^FROM ' "$containerfile")" -eq 1 ] ||
  fail 'Containerfile must have exactly one base stage'
assert_line 'FROM ${BASE_IMAGE}'
assert_line 'COPY --chown=0:0 rootfs/ /'
assert_line 'COPY --chown=0:0 metadata/ /usr/share/lunaflux/'
assert_line 'LABEL io.lunaflux.labels.authority="diagnostic-only"'
assert_line 'USER 65532:65532'
assert_line 'HEALTHCHECK NONE'
assert_line 'ENTRYPOINT ["/opt/lunaflux/bin/lunaflux"]'
[ "$(grep -c '^COPY ' "$containerfile")" -eq 2 ] ||
  fail 'Containerfile must copy only the verified rootfs and metadata trees'
if grep -E '^[[:space:]]*(RUN|CMD|SHELL|ADD|ENV|VOLUME)[[:space:]]' \
  "$containerfile" >/dev/null 2>&1; then
  fail 'Containerfile contains a mutable build or runtime directive'
fi
if grep -E '^ARG BASE_IMAGE=' "$containerfile" >/dev/null 2>&1; then
  fail 'Containerfile supplies a BASE_IMAGE fallback'
fi
if grep -E -i '^#[[:space:]]*syntax=' "$containerfile" >/dev/null 2>&1; then
  fail 'Containerfile must not select a mutable external frontend'
fi
if grep -E -i '^[[:space:]]*(RUN|CMD|ENTRYPOINT).*(python|pip|apt|apk|dnf|yum|gcc|clang|nvcc|nvrtc|ptx|moon|bash|/bin/sh)' \
  "$containerfile" >/dev/null 2>&1; then
  fail 'Containerfile exposes a shell, package manager, compiler, Python, or JIT path'
fi

temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-oci-gate.XXXXXX")
temporary_root=$(CDPATH= cd -- "$temporary_root" && pwd -P)
trap 'chmod -R u+w "$temporary_root" 2>/dev/null || true; rm -rf "$temporary_root"' EXIT HUP INT TERM
valid=$temporary_root/valid
mkdir -p "$valid/rootfs/opt/lunaflux/bin" \
  "$valid/rootfs/opt/lunaflux/lib" \
  "$valid/rootfs/opt/lunaflux/kernels" \
  "$valid/rootfs/usr/share/licenses/lunaflux" \
  "$valid/rootfs/var/lib/lunaflux/model" "$valid/metadata"

# Deliberately synthetic, non-published fixture identity; never a deployment base.
fixture_digest=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
base_image=registry.invalid/lunaflux-gate/cuda-runtime@sha256:$fixture_digest
for binary in lunaflux lunaflux-device-worker; do
  printf '\177ELF\002\001\001\000\000\000\000\000\000\000\000\000\002\000\076\000fixture\n' \
    > "$valid/rootfs/opt/lunaflux/bin/$binary"
  chmod 555 "$valid/rootfs/opt/lunaflux/bin/$binary"
done
printf '%s\n' 'external-read-only-model-root-required' \
  > "$valid/rootfs/var/lib/lunaflux/model/.mount-contract"
printf '%s\n' 'synthetic-aot-module' \
  > "$valid/rootfs/opt/lunaflux/kernels/gate.cubin"
module_digest=$(sha256_file "$valid/rootfs/opt/lunaflux/kernels/gate.cubin")
printf '{"modules":[{"path":"gate.cubin","sha256":"%s"}]}\n' \
  "$module_digest" > "$valid/rootfs/opt/lunaflux/kernels/execution.json"
manifest_digest=$(sha256_file \
  "$valid/rootfs/opt/lunaflux/kernels/execution.json")
printf '%s\n' "$base_image" > "$valid/metadata/base-image.ref"
printf '%s\n' 'execution.json' > "$valid/metadata/kernel-manifest.relative"
printf '%s\n' "$manifest_digest" > "$valid/metadata/kernel-manifest.sha256"
printf '%s\n' 'x86_64' > "$valid/metadata/linux-architecture"
printf '%s\n' 'none' > "$valid/metadata/runtime-libraries.list"
cp "$repo_root/LICENSE" \
  "$valid/rootfs/usr/share/licenses/lunaflux/LICENSE"
"$license_assembler" "$valid/metadata/license-inventory.json" >/dev/null
: > "$valid/metadata/artifacts.sha256"
chmod 444 "$valid/rootfs/var/lib/lunaflux/model/.mount-contract" \
  "$valid/rootfs/usr/share/licenses/lunaflux/LICENSE" \
  "$valid/rootfs/opt/lunaflux/kernels/gate.cubin" \
  "$valid/rootfs/opt/lunaflux/kernels/execution.json" "$valid/metadata/"*
find "$valid/rootfs" "$valid/metadata" -type d -exec chmod 555 {} \;
write_inventory "$valid"

"$verifier" "$base_image" "$valid" >/dev/null
assert_rejected unpinned-tag registry.invalid/cuda-runtime:12.8 "$valid"
assert_rejected mutable-tag-with-digest \
  registry.invalid/cuda-runtime:12.8@sha256:$fixture_digest "$valid"
assert_rejected metadata-base-mismatch \
  registry.invalid/another-runtime@sha256:$fixture_digest "$valid"

repeat_inventory=$temporary_root/license-inventory.json
"$license_assembler" "$repeat_inventory" >/dev/null
cmp -s "$repeat_inventory" "$valid/metadata/license-inventory.json" ||
  fail 'product license inventory is not deterministic'
if "$license_assembler" "$repeat_inventory" >/dev/null 2>&1; then
  fail 'product license inventory assembler overwrote an existing output'
fi
broken_inventory=$temporary_root/broken-license-inventory.json
ln -s "$temporary_root/absent-license-inventory" "$broken_inventory"
if "$license_assembler" "$broken_inventory" >/dev/null 2>&1; then
  fail 'product license inventory assembler replaced a broken symlink output'
fi

missing_license=$temporary_root/missing-license
cp -R "$valid" "$missing_license"
chmod 755 "$missing_license/rootfs/usr/share/licenses/lunaflux"
rm "$missing_license/rootfs/usr/share/licenses/lunaflux/LICENSE"
chmod 555 "$missing_license/rootfs/usr/share/licenses/lunaflux"
write_inventory "$missing_license"
assert_rejected missing-product-license "$base_image" "$missing_license"

mutated_license=$temporary_root/mutated-license
cp -R "$valid" "$mutated_license"
chmod 644 "$mutated_license/rootfs/usr/share/licenses/lunaflux/LICENSE"
printf '%s\n' 'substituted license text' \
  > "$mutated_license/rootfs/usr/share/licenses/lunaflux/LICENSE"
chmod 444 "$mutated_license/rootfs/usr/share/licenses/lunaflux/LICENSE"
write_inventory "$mutated_license"
assert_rejected substituted-product-license "$base_image" "$mutated_license"

substituted_inventory=$temporary_root/substituted-license-inventory
cp -R "$valid" "$substituted_inventory"
chmod 644 "$substituted_inventory/metadata/license-inventory.json"
sed 's/Apache-2.0/MIT/' "$valid/metadata/license-inventory.json" \
  > "$substituted_inventory/metadata/license-inventory.json"
chmod 444 "$substituted_inventory/metadata/license-inventory.json"
assert_rejected substituted-license-inventory "$base_image" \
  "$substituted_inventory"

writable=$temporary_root/writable
cp -R "$valid" "$writable"
chmod 644 "$writable/rootfs/opt/lunaflux/bin/lunaflux"
assert_rejected writable-payload "$base_image" "$writable"

ambient=$temporary_root/ambient
cp -R "$valid" "$ambient"
chmod 755 "$ambient/metadata"
printf '%s\n' ambient > "$ambient/metadata/unlisted"
chmod 444 "$ambient/metadata/unlisted"
chmod 555 "$ambient/metadata"
assert_rejected ambient-metadata "$base_image" "$ambient"

hardlinked=$temporary_root/hardlinked
cp -R "$valid" "$hardlinked"
ln "$hardlinked/rootfs/opt/lunaflux/kernels/gate.cubin" \
  "$temporary_root/gate.cubin.alias"
assert_rejected hard-linked-payload "$base_image" "$hardlinked"
rm "$temporary_root/gate.cubin.alias"

jit=$temporary_root/jit
cp -R "$valid" "$jit"
chmod 755 "$jit/rootfs/opt/lunaflux/kernels"
printf '%s\n' 'runtime source forbidden' \
  > "$jit/rootfs/opt/lunaflux/kernels/late.ptx"
chmod 444 "$jit/rootfs/opt/lunaflux/kernels/late.ptx"
chmod 555 "$jit/rootfs/opt/lunaflux/kernels"
write_inventory "$jit"
assert_rejected ptx-payload "$base_image" "$jit"

toolchain=$temporary_root/toolchain
cp -R "$valid" "$toolchain"
chmod 755 "$toolchain/rootfs/opt/lunaflux/lib"
printf '%s\n' 'synthetic forbidden runtime JIT library' \
  > "$toolchain/rootfs/opt/lunaflux/lib/libnvrtc.so"
chmod 444 "$toolchain/rootfs/opt/lunaflux/lib/libnvrtc.so"
chmod 555 "$toolchain/rootfs/opt/lunaflux/lib"
write_inventory "$toolchain"
assert_rejected runtime-jit-library "$base_image" "$toolchain"

unbound=$temporary_root/unbound
cp -R "$valid" "$unbound"
chmod 644 "$unbound/rootfs/opt/lunaflux/kernels/execution.json"
printf '%s\n' '{"modules":[]}' \
  > "$unbound/rootfs/opt/lunaflux/kernels/execution.json"
chmod 444 "$unbound/rootfs/opt/lunaflux/kernels/execution.json"
manifest_digest=$(sha256_file \
  "$unbound/rootfs/opt/lunaflux/kernels/execution.json")
chmod 644 "$unbound/metadata/kernel-manifest.sha256"
printf '%s\n' "$manifest_digest" \
  > "$unbound/metadata/kernel-manifest.sha256"
chmod 444 "$unbound/metadata/kernel-manifest.sha256"
write_inventory "$unbound"
assert_rejected unbound-aot-module "$base_image" "$unbound"

wrong_path=$temporary_root/wrong-path
cp -R "$valid" "$wrong_path"
chmod 644 "$wrong_path/rootfs/opt/lunaflux/kernels/execution.json"
printf '{"modules":[{"path":"other.cubin","sha256":"%s"}]}\n' \
  "$module_digest" \
  > "$wrong_path/rootfs/opt/lunaflux/kernels/execution.json"
chmod 444 "$wrong_path/rootfs/opt/lunaflux/kernels/execution.json"
manifest_digest=$(sha256_file \
  "$wrong_path/rootfs/opt/lunaflux/kernels/execution.json")
chmod 644 "$wrong_path/metadata/kernel-manifest.sha256"
printf '%s\n' "$manifest_digest" \
  > "$wrong_path/metadata/kernel-manifest.sha256"
chmod 444 "$wrong_path/metadata/kernel-manifest.sha256"
write_inventory "$wrong_path"
assert_rejected wrong-module-path-with-retained-digest "$base_image" \
  "$wrong_path"

missing=$temporary_root/missing
cp -R "$valid" "$missing"
chmod 755 "$missing/rootfs/opt/lunaflux/kernels"
rm "$missing/rootfs/opt/lunaflux/kernels/execution.json"
chmod 555 "$missing/rootfs/opt/lunaflux/kernels"
write_inventory "$missing"
assert_rejected missing-manifest "$base_image" "$missing"

printf '%s\n' 'LunaFlux OCI packaging static and hostile-context gates passed.'
