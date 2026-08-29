#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
assembler=$repo_root/scripts/assemble-release-bundle.sh
verifier=$repo_root/scripts/verify-release-bundle.sh

fail() {
  printf '%s\n' "LunaFlux release-bundle assembly gate failed: $1" >&2
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

write_inventory() {
  root=$1
  output=$2
  find "$root" -type f -print | sed "s#^$root/##" | LC_ALL=C sort |
    while IFS= read -r relative; do
      printf '%s  %s\n' "$(sha256_file "$root/$relative")" "$relative"
    done > "$output"
}

assert_assembly_rejected() {
  label=$1
  input=$2
  output=$3
  digest=$(sha256_file "$input")
  if "$assembler" "$input#sha256=$digest" "$output" >/dev/null 2>&1; then
    fail "assembler passed hostile fixture: $label"
  fi
  [ ! -e "$output" ] || fail "failed assembly retained output: $label"
}

assert_verifier_rejected() {
  label=$1
  bundle=$2
  if "$verifier" "$bundle" >/dev/null 2>&1; then
    fail "verifier passed hostile fixture: $label"
  fi
}

sh -n "$repo_root/scripts/release-bundle-common.sh"
sh -n "$assembler"
sh -n "$verifier"

fixture_root=$(mktemp -d /tmp/lunaflux-release-bundle-gate.XXXXXX)
fixture_root=$(CDPATH= cd -- "$fixture_root" && pwd -P)
trap 'chmod -R u+w "$fixture_root" 2>/dev/null || true; rm -rf "$fixture_root"' EXIT HUP INT TERM
sources=$fixture_root/sources
mkdir -p "$sources/bin" "$sources/model/runtime" "$sources/policy/instance" \
  "$sources/kernels/sha256" "$sources/inventories"

# Tiny ELF-shaped files exercise packaging only. They are not executable
# inference engines, kernel correctness evidence, or release artifacts.
for binary in lunaflux lunaflux-device-worker; do
  printf '\177ELF\002\001\001\000\000\000\000\000\000\000\000\000\002\000\076\000synthetic-fixture-only\n' \
    > "$sources/bin/$binary"
done
printf '%s\n' '{"fixture":"config-only"}' > "$sources/model/config.json"
printf '%s\n' '{"fixture":"tokenizer-only"}' > "$sources/model/tokenizer.json"
printf '%s\n' 'synthetic-numeric-weight-bytes-not-a-model' \
  > "$sources/model/model.numeric.safetensors"
printf '%s\n' '{"schema_version":"lunaflux.runtime.i8.v2","fixture_only":true}' \
  > "$sources/model/runtime/descriptor.json"
descriptor_sha=$(sha256_file "$sources/model/runtime/descriptor.json")
printf '%s\n' '{"schema_version":"lunaflux.instance-policy.v1","fixture_only":true}' \
  > "$sources/policy/instance/policy.json"
policy_sha=$(sha256_file "$sources/policy/instance/policy.json")
printf '%s\n' 'synthetic-aot-module-not-a-production-kernel' \
  > "$sources/kernels/sha256/fixture.cubin"
module_sha=$(sha256_file "$sources/kernels/sha256/fixture.cubin")
printf '{"modules":[{"path":"sha256/fixture.cubin","sha256":"%s"}]}\n' \
  "$module_sha" > "$sources/kernels/execution-i8-v4.json"
kernel_manifest_sha=$(sha256_file "$sources/kernels/execution-i8-v4.json")

lunaflux_sha=$(sha256_file "$sources/bin/lunaflux")
worker_sha=$(sha256_file "$sources/bin/lunaflux-device-worker")
cat > "$sources/launch.json" <<EOF
{"schema":"lunaflux.launch.v2","runtime_recipe":"dense_llama_i8_paged_aot_v6","model_root":"/var/lib/lunaflux/model","kernel_root":"/opt/lunaflux/kernels","policy_root":"/var/lib/lunaflux/policy","runtime_descriptor":{"locator":"runtime/descriptor.json","sha256":"$descriptor_sha"},"instance_policy":{"locator":"instance/policy.json","sha256":"$policy_sha"},"worker_executable":{"path":"/opt/lunaflux/bin/lunaflux-device-worker","sha256":"$worker_sha"},"luna_approval":{"mode":"none"}}
EOF
launch_sha=$(sha256_file "$sources/launch.json")

model_inventory=$sources/inventories/model.files.sha256
policy_inventory=$sources/inventories/policy.files.sha256
kernel_inventory=$sources/inventories/kernel.files.sha256
library_inventory=$sources/inventories/runtime-libraries.files.sha256
write_inventory "$sources/model" "$model_inventory"
write_inventory "$sources/policy" "$policy_inventory"
write_inventory "$sources/kernels" "$kernel_inventory"
printf '%s\n' none > "$library_inventory"
model_inventory_sha=$(sha256_file "$model_inventory")
policy_inventory_sha=$(sha256_file "$policy_inventory")
kernel_inventory_sha=$(sha256_file "$kernel_inventory")
library_inventory_sha=$(sha256_file "$library_inventory")

base_digest=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
base_image=registry.invalid/lunaflux-fixture/cuda@sha256:$base_digest
input=$fixture_root/assembly-input.v1
cat > "$input" <<EOF
schema=lunaflux.deployment-assembly.v1
base_image=$base_image
linux_architecture=x86_64
lunaflux_source=$sources/bin/lunaflux
lunaflux_sha256=$lunaflux_sha
worker_source=$sources/bin/lunaflux-device-worker
worker_sha256=$worker_sha
launch_source=$sources/launch.json
launch_sha256=$launch_sha
model_source_root=$sources/model
model_inventory_source=$model_inventory
model_inventory_sha256=$model_inventory_sha
runtime_descriptor_relative=runtime/descriptor.json
runtime_descriptor_sha256=$descriptor_sha
policy_source_root=$sources/policy
policy_inventory_source=$policy_inventory
policy_inventory_sha256=$policy_inventory_sha
instance_policy_relative=instance/policy.json
instance_policy_sha256=$policy_sha
kernel_source_root=$sources/kernels
kernel_inventory_source=$kernel_inventory
kernel_inventory_sha256=$kernel_inventory_sha
kernel_manifest_relative=execution-i8-v4.json
kernel_manifest_sha256=$kernel_manifest_sha
runtime_library_source_root=none
runtime_library_inventory_source=$library_inventory
runtime_library_inventory_sha256=$library_inventory_sha
EOF

input_sha=$(sha256_file "$input")
valid=$fixture_root/valid-bundle
"$assembler" "$input#sha256=$input_sha" "$valid" >/dev/null
"$verifier" "$valid" >/dev/null

if "$assembler" "$input#sha256=$input_sha" "$valid" >/dev/null 2>&1; then
  fail 'assembler overwrote an existing output'
fi
"$verifier" "$valid" >/dev/null || fail 'overwrite attempt changed completed bundle'

# Fail the second transfer after one staged entry has moved. The retained
# claim must still authorize cleanup of the otherwise partial output.
fake_bin=$fixture_root/fake-bin
mkdir "$fake_bin"
fake_mv_state=$fixture_root/fake-mv-state
cat > "$fake_bin/mv" <<EOF
#!/bin/sh
count=0
[ ! -f "$fake_mv_state" ] || count=\$(sed -n '1p' "$fake_mv_state")
count=\$((count + 1))
printf '%s\n' "\$count" > "$fake_mv_state"
[ "\$count" -ne 2 ] || exit 73
exec /bin/mv "\$@"
EOF
chmod 755 "$fake_bin/mv"
transfer_failure=$fixture_root/transfer-failure-output
if PATH="$fake_bin:$PATH" "$assembler" "$input#sha256=$input_sha" \
  "$transfer_failure" >/dev/null 2>&1; then
  fail 'injected transfer failure unexpectedly succeeded'
fi
[ ! -e "$transfer_failure" ] ||
  fail 'injected transfer failure retained a partial output'

substituted_root=$fixture_root/substituted-sources
cp -R "$sources" "$substituted_root"
printf '%s\n' substituted >> "$substituted_root/model/config.json"
substituted_input=$fixture_root/substituted-input.v1
sed "s#$sources#$substituted_root#g" "$input" > "$substituted_input"
assert_assembly_rejected substituted-payload "$substituted_input" \
  "$fixture_root/substituted-output"

extra_root=$fixture_root/extra-sources
cp -R "$sources" "$extra_root"
printf '%s\n' '{}' > "$extra_root/model/ambient.json"
extra_input=$fixture_root/extra-input.v1
sed "s#$sources#$extra_root#g" "$input" > "$extra_input"
assert_assembly_rejected unlisted-extra-file "$extra_input" "$fixture_root/extra-output"

extra_bundle=$fixture_root/extra-bundle
cp -R "$valid" "$extra_bundle"
chmod 755 "$extra_bundle/evidence"
printf '%s\n' ambient > "$extra_bundle/evidence/ambient.txt"
chmod 444 "$extra_bundle/evidence/ambient.txt"
chmod 555 "$extra_bundle/evidence"
assert_verifier_rejected extra-staged-file "$extra_bundle"

substituted_bundle=$fixture_root/substituted-bundle
cp -R "$valid" "$substituted_bundle"
chmod 644 "$substituted_bundle/model-root/config.json"
printf '%s\n' substituted >> "$substituted_bundle/model-root/config.json"
chmod 444 "$substituted_bundle/model-root/config.json"
assert_verifier_rejected substituted-staged-file "$substituted_bundle"

jit_root=$fixture_root/jit-sources
cp -R "$sources" "$jit_root"
printf '%s\n' 'synthetic runtime source' > "$jit_root/kernels/late.ptx"
write_inventory "$jit_root/kernels" "$jit_root/inventories/kernel.files.sha256"
jit_input=$fixture_root/jit-input.v1
sed "s#$sources#$jit_root#g" "$input" |
  sed "s#kernel_inventory_sha256=$kernel_inventory_sha#kernel_inventory_sha256=$(sha256_file "$jit_root/inventories/kernel.files.sha256")#" \
  > "$jit_input"
assert_assembly_rejected runtime-ptx "$jit_input" "$fixture_root/jit-output"

printf '%s\n' 'LunaFlux deterministic deployment-bundle assembly gates passed.'
