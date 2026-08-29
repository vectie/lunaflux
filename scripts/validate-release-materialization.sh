#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
materializer=$repo_root/scripts/materialize-release-bundle.sh
recovery=$repo_root/scripts/recover-release-materialization.sh
verifier=$repo_root/scripts/verify-materialized-release-bundle.sh

fail() {
  printf '%s\n' "LunaFlux release-materialization gate failed: $1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
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

assert_rejected_without_output() {
  label=$1
  input_arg=$2
  output=$3
  tool_arg=$4
  if "$materializer" "$input_arg" "$output" "$tool_arg" >/dev/null 2>&1; then
    fail "hostile materialization passed: $label"
  fi
  [ ! -e "$output" ] || fail "hostile materialization retained output: $label"
}

sh -n "$materializer"
sh -n "$recovery"
sh -n "$verifier"

fixture=$(mktemp -d /tmp/lunaflux-release-materialization-gate.XXXXXX)
fixture=$(CDPATH= cd -- "$fixture" && pwd -P)
trap 'chmod -R u+w "$fixture" 2>/dev/null || true; rm -rf "$fixture"' EXIT HUP INT TERM
sources=$fixture/sources
mkdir -p "$sources/bin" "$sources/model/runtime" "$sources/policy/instance" \
  "$sources/kernels/sha256" "$sources/inventories"

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
input=$fixture/assembly-input.v1
cat > "$input" <<EOF
schema=lunaflux.deployment-assembly.v1
base_image=registry.invalid/lunaflux-fixture/cuda@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
linux_architecture=x86_64
lunaflux_source=$sources/bin/lunaflux
lunaflux_sha256=$lunaflux_sha
worker_source=$sources/bin/lunaflux-device-worker
worker_sha256=$worker_sha
launch_source=$sources/launch.json
launch_sha256=$launch_sha
model_source_root=$sources/model
model_inventory_source=$model_inventory
model_inventory_sha256=$(sha256_file "$model_inventory")
runtime_descriptor_relative=runtime/descriptor.json
runtime_descriptor_sha256=$descriptor_sha
policy_source_root=$sources/policy
policy_inventory_source=$policy_inventory
policy_inventory_sha256=$(sha256_file "$policy_inventory")
instance_policy_relative=instance/policy.json
instance_policy_sha256=$policy_sha
kernel_source_root=$sources/kernels
kernel_inventory_source=$kernel_inventory
kernel_inventory_sha256=$(sha256_file "$kernel_inventory")
kernel_manifest_relative=execution-i8-v4.json
kernel_manifest_sha256=$kernel_manifest_sha
runtime_library_source_root=none
runtime_library_inventory_source=$library_inventory
runtime_library_inventory_sha256=$(sha256_file "$library_inventory")
EOF
input_arg=$input#sha256=$(sha256_file "$input")

tool_evidence=$fixture/tool-evidence
tool=$fixture/lunaflux-host-preflight
cat > "$tool" <<EOF
#!/bin/sh
set -eu
[ "\$#" -eq 2 ] && [ "\$1" = validate-materialized-release ] || exit 2
bundle_sha=\${2##*#sha256=}
cat > "$tool_evidence" <<EVIDENCE
schema=lunaflux-release-materialization-preflight.v1
runtime_recipe=dense_llama_i8_paged_aot_v6
deployment_bundle_sha256=\$bundle_sha
launch_sha256=$launch_sha
runtime_descriptor_sha256=$descriptor_sha
instance_policy_sha256=$policy_sha
tokenizer_sha256=1111111111111111111111111111111111111111111111111111111111111111
worker_executable_sha256=$worker_sha
model_content_sha256=2222222222222222222222222222222222222222222222222222222222222222
model_plan_sha256=3333333333333333333333333333333333333333333333333333333333333333
bootstrap_sha256=4444444444444444444444444444444444444444444444444444444444444444
bootstrap_source_sha256=5555555555555555555555555555555555555555555555555555555555555555
device_ordinal=0
compute_major=8
compute_minor=9
source_target_binding=1
semantic_join=1
filesystem_authority_closed=1
device_opened=0
compiler_jit_authority=0
EVIDENCE
cat "$tool_evidence"
printf '\n'
if command -v sha256sum >/dev/null 2>&1; then
  digest=\$(sha256sum "$tool_evidence" | awk '{print \$1}')
else
  digest=\$(shasum -a 256 "$tool_evidence" | awk '{print \$1}')
fi
printf '%s\n' "release_materialization_preflight_sha256=\$digest"
EOF
chmod 755 "$tool"
tool_arg=$tool#sha256=$(sha256_file "$tool")
approved_tool_arg=$tool_arg

make_crashed_publication() {
  crash_prefix=$1
  crash_output=$2
  crash_bin=$fixture/crash-bin-$crash_prefix-$(basename -- "$crash_output")
  crash_state=$fixture/crash-state-$crash_prefix-$(basename -- "$crash_output")
  mkdir "$crash_bin"
  cat > "$crash_bin/mv" <<EOF
#!/bin/sh
set -eu
case "\$1" in
  "$crash_output/.bundle/"*)
    count=0
    [ ! -f "$crash_state" ] || count=\$(sed -n '1p' "$crash_state")
    if [ "\$count" -eq "$crash_prefix" ] && [ "$crash_prefix" -lt 6 ]; then
      kill -KILL "\$PPID"
      exit 137
    fi
    /bin/mv "\$@"
    count=\$((count + 1))
    printf '%s\n' "\$count" > "$crash_state"
    ;;
  *) exec /bin/mv "\$@" ;;
esac
EOF
  cat > "$crash_bin/rmdir" <<EOF
#!/bin/sh
set -eu
if [ "$crash_prefix" -eq 6 ] && [ "\$#" -eq 1 ] &&
  [ "\$1" = "$crash_output/.bundle" ]; then
  kill -KILL "\$PPID"
  exit 137
fi
exec /bin/rmdir "\$@"
EOF
  chmod 755 "$crash_bin/mv" "$crash_bin/rmdir"
  if PATH="$crash_bin:$PATH" "$materializer" "$input_arg" "$crash_output" \
    "$approved_tool_arg" >/dev/null 2>&1; then
    fail "publication crash point unexpectedly completed: $crash_prefix"
  fi
  [ -d "$crash_output" ] && [ ! -L "$crash_output" ] &&
    [ -f "$crash_output/.materialization-claim" ] ||
    fail "publication crash point lost its exact claim: $crash_prefix"
}

assert_recovery_refused() {
  recovery_label=$1
  recovery_output=$2
  if "$recovery" "$recovery_output" >/dev/null 2>&1; then
    fail "hostile recovery passed: $recovery_label"
  fi
  [ -f "$recovery_output/.materialization-claim" ] ||
    fail "hostile recovery mutated the exact claim: $recovery_label"
}

valid=$fixture/valid
"$materializer" "$input_arg" "$valid" "$tool_arg" >/dev/null
"$verifier" "$valid" >/dev/null
if "$materializer" "$input_arg" "$valid" "$tool_arg" >/dev/null 2>&1; then
  fail 'materializer overwrote a completed output'
fi
"$verifier" "$valid" >/dev/null || fail 'overwrite attempt changed output'

failing_tool=$fixture/failing-preflight
printf '%s\n' '#!/bin/sh' 'exit 73' > "$failing_tool"
chmod 755 "$failing_tool"
assert_rejected_without_output semantic-preflight-failure "$input_arg" \
  "$fixture/preflight-failure" \
  "$failing_tool#sha256=$(sha256_file "$failing_tool")"

assert_rejected_without_output tool-digest-substitution "$input_arg" \
  "$fixture/tool-substitution" \
  "$tool#sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

bad_tool=$fixture/substituting-preflight
sed "s/worker_executable_sha256=$worker_sha/worker_executable_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff/" \
  "$tool" > "$bad_tool"
chmod 755 "$bad_tool"
assert_rejected_without_output semantic-substitution "$input_arg" \
  "$fixture/semantic-substitution" \
  "$bad_tool#sha256=$(sha256_file "$bad_tool")"

# Every exact prefix transition, including the empty stage after transfer six,
# must remain explicitly recoverable after an untrappable process death.
for crash_prefix in 0 1 2 3 4 5 6; do
  crash_output=$fixture/crash-prefix-$crash_prefix
  make_crashed_publication "$crash_prefix" "$crash_output"
  if [ "$crash_prefix" -eq 0 ]; then
    rm "$crash_output/.materialization-prepared"
  fi
  "$recovery" "$crash_output" >/dev/null
  "$verifier" "$crash_output" >/dev/null ||
    fail "recovered publication is not exact: $crash_prefix"
done

# CLAIMED cleanup unlinks only the exact mode-400 regular claim and uses rmdir
# on the otherwise empty current-user-owned output.
claimed_only=$fixture/claimed-only
mkdir "$claimed_only"
cat > "$claimed_only/.materialization-claim" <<EOF
schema=lunaflux.release-materialization-claim.v2
state=claimed
output_path=$claimed_only
owner_uid=$(id -u)
assembly_input_sha256=${input_arg##*#sha256=}
semantic_preflight_tool_sha256=${approved_tool_arg##*#sha256=}
materializer_sha256=$(sha256_file "$materializer")
publication_order=bundle.files.sha256,evidence,launch-root,model-root,oci-context,policy-root
EOF
chmod 400 "$claimed_only/.materialization-claim"
"$recovery" "$claimed_only" >/dev/null
[ ! -e "$claimed_only" ] || fail 'empty exact claim recovery retained output'

# Ambiguity is refusal-only: no recovery path may delete or move content after
# an identity, namespace, ordering, link, type, or inventory substitution.
hostile=$fixture/hostile-recovery
make_crashed_publication 2 "$hostile"
ln "$hostile/.materialization-claim" "$fixture/claim-alias"
assert_recovery_refused claim-hard-link "$hostile"
rm "$fixture/claim-alias"
touch "$hostile/unexpected"
assert_recovery_refused unknown-entry "$hostile"
rm "$hostile/unexpected"
ln -s "$hostile/.bundle" "$hostile/substituted-link"
assert_recovery_refused symlink-substitution "$hostile"
rm "$hostile/substituted-link"
mkfifo "$hostile/substituted-fifo"
assert_recovery_refused special-file-substitution "$hostile"
rm "$hostile/substituted-fifo"
mv "$hostile/.bundle/model-root" "$hostile/model-root"
assert_recovery_refused nonprefix-publication "$hostile"
mv "$hostile/model-root" "$hostile/.bundle/model-root"
cloned=$fixture/cloned-claim
cp -R "$hostile" "$cloned"
assert_recovery_refused canonical-output-substitution "$cloned"
chmod 644 "$hostile/.bundle/launch-root/lunaflux.launch.json"
printf '%s\n' substituted >> "$hostile/.bundle/launch-root/lunaflux.launch.json"
assert_recovery_refused inventory-substitution "$hostile"

tampered=$fixture/tampered
cp -R "$valid" "$tampered"
chmod 755 "$tampered/evidence"
chmod 644 "$tampered/evidence/release-materialization-preflight.v1"
sed -i.bak 's/source_target_binding=1/source_target_binding=0/' \
  "$tampered/evidence/release-materialization-preflight.v1"
rm "$tampered/evidence/release-materialization-preflight.v1.bak"
chmod 444 "$tampered/evidence/release-materialization-preflight.v1"
chmod 555 "$tampered/evidence"
if "$verifier" "$tampered" >/dev/null 2>&1; then
  fail 'verifier admitted substituted semantic evidence'
fi

printf '%s\n' 'LunaFlux atomic release-materialization gates passed.'
