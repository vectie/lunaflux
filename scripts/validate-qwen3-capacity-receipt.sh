#!/bin/sh
set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf 'Qwen3 capacity receipt boundary failed: %s\n' "$1" >&2
  exit 1
}

materializer=scripts/materialize-qwen3-authenticated-capacity.sh
verifier=scripts/verify-qwen3-authenticated-capacity.sh
common=scripts/qwen3-capacity-receipt-common.sh
launch_materializer=scripts/materialize-qwen3-bf16-v12-launch.sh
for script in "$materializer" "$verifier" "$common" "$launch_materializer"; do
  sh -n "$script" || fail "shell syntax is invalid: $script"
done
for anchor in \
  'native correctness profile requires authenticated c1 release geometry' \
  'native-framed-c32-benchmark-v1|openai-responses-v1' \
  'benchmark profile requires authenticated c32 release geometry'; do
  grep -F "$anchor" "$launch_materializer" >/dev/null ||
    fail "Qwen native c1/c32 profile anchor is absent: $anchor"
done

scratch=$(mktemp -d /tmp/lunaflux-qwen3-capacity-test.XXXXXX) ||
  fail 'could not create fixture scratch'
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
trap 'chmod -R u+rwX "$scratch" 2>/dev/null || true; rm -rf -- "$scratch"' EXIT HUP INT TERM
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | sed 's/[[:space:]].*$//'
  else
    shasum -a 256 "$1" | sed 's/[[:space:]].*$//'
  fi
}
digest_a=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
digest_b=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
digest_c=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
digest_d=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
digest_e=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
bind=$scratch/release-bind.v1
write_bind() {
  bind_rows=$1
  bind_tokens=$2
  {
    printf '%s\n' 'schema=lunaflux-qwen3-bf16-release-bind.v1'
    printf '%s\n' 'recipe=dense_qwen3_bf16_paged_aot_v12'
    printf 'model_content_sha256=%s\n' "$digest_a"
    printf 'weight_route_manifest_sha256=%s\n' "$digest_b"
    printf 'model_plan_sha256=%s\n' "$digest_c"
    printf '%s\n' 'target=sm_120'
    printf 'max_batch_rows=%s\n' "$bind_rows"
    printf 'max_query_rows=%s\n' "$bind_rows"
    printf 'max_query_tokens=%s\n' "$bind_tokens"
    printf 'kernel_manifest_sha256=%s\n' "$digest_d"
    printf 'admitted_bootstrap_sha256=%s\n' "$digest_e"
    printf '%s\n' 'compiler_invoked=0'
    printf '%s\n' 'device_opened=0'
    printf '%s\n' 'runtime_authority=0'
  } >"$bind"
}
write_bind 32 256

deployment=$scratch/deployment
mkdir -p "$deployment/evidence" "$deployment/model-root/runtime" \
  "$deployment/policy-root/instance"
printf '%s\n' '{"schema":"lunaflux.launch.v5","runtime_recipe":"dense_qwen3_bf16_paged_aot_v12"}' \
  >"$deployment/lunaflux.launch.json"
cp "$bind" "$deployment/evidence/release-bind.v1"
printf '%s\n' '{"schema_version":"lunaflux.runtime.qwen3_bf16.v1","model":{"max_batch_rows":32}}' \
  >"$deployment/model-root/runtime/descriptor.json"
printf '%s\n' '{"schema_version":"lunaflux.instance-policy.v1","scheduler":{"max_active_requests":32,"max_waiting_requests":32}}' \
  >"$deployment/policy-root/instance/policy.json"
runtime=$scratch/lunaflux
bridge=$scratch/qwen3-bridge
printf '%s\n' runtime >"$runtime"
printf '%s\n' bridge >"$bridge"
chmod 0555 "$runtime" "$bridge"

bind_sha=$(sha256_file "$bind")
launch_sha=$(sha256_file "$deployment/lunaflux.launch.json")
runtime_sha=$(sha256_file "$runtime")
bridge_sha=$(sha256_file "$bridge")
receipt=$scratch/capacity.json
"$materializer" "$bind#sha256=$bind_sha" \
  "$deployment#sha256=$launch_sha" "$runtime#sha256=$runtime_sha" \
  "$bridge#sha256=$bridge_sha" "$receipt" >"$scratch/materialize.stdout"
receipt_sha=$(sha256_file "$receipt")
"$verifier" "$bind#sha256=$bind_sha" \
  "$deployment#sha256=$launch_sha" "$runtime#sha256=$runtime_sha" \
  "$bridge#sha256=$bridge_sha" "$receipt#sha256=$receipt_sha" \
  >"$scratch/verify.stdout"
grep -Fx 'outcome=passed' "$scratch/verify.stdout" >/dev/null ||
  fail 'positive receipt did not verify'
mode=$(stat -f '%Lp' "$receipt" 2>/dev/null || stat -c '%a' "$receipt")
[ "$mode" = 444 ] || fail 'published receipt is not read-only'
independent_auth=$scratch/independent-authentication.v1
{
  printf '%s\n' 'schema=lunaflux.qwen3-authenticated-capacity-authentication.v1'
  printf 'release_bind_stdout_sha256=%s\n' "$bind_sha"
  printf 'launch_sha256=%s\n' "$launch_sha"
  printf 'model_content_sha256=%s\n' "$digest_a"
  printf 'model_plan_sha256=%s\n' "$digest_c"
  printf 'weight_route_manifest_sha256=%s\n' "$digest_b"
  printf 'kernel_manifest_sha256=%s\n' "$digest_d"
  printf 'admitted_bootstrap_sha256=%s\n' "$digest_e"
  printf '%s\n' 'max_batch_rows=32'
  printf '%s\n' 'max_query_rows=32'
  printf '%s\n' 'max_query_tokens=256'
  printf '%s\n' 'max_concurrency=32'
  printf 'runtime_executable_sha256=%s\n' "$runtime_sha"
  printf 'token_id_sse_bridge_sha256=%s\n' "$bridge_sha"
} >"$independent_auth"
independent_auth_sha=$(sha256_file "$independent_auth")
grep -F "\"authentication_sha256\":\"$independent_auth_sha\"" "$receipt" >/dev/null ||
  fail 'receipt authentication digest is not independently reproducible'

cp "$deployment/model-root/runtime/descriptor.json" "$scratch/descriptor.c32"
sed 's/"max_batch_rows":32/"max_batch_rows":320/' \
  "$scratch/descriptor.c32" >"$deployment/model-root/runtime/descriptor.json"
if "$materializer" "$bind#sha256=$bind_sha" \
  "$deployment#sha256=$launch_sha" "$runtime#sha256=$runtime_sha" \
  "$bridge#sha256=$bridge_sha" "$scratch/c320.json" >/dev/null 2>&1; then
  fail 'descriptor capacity 320 was mistaken for exact c32'
fi
cp "$scratch/descriptor.c32" "$deployment/model-root/runtime/descriptor.json"

write_bind 1 1
c1_sha=$(sha256_file "$bind")
cp "$bind" "$deployment/evidence/release-bind.v1"
if "$materializer" "$bind#sha256=$c1_sha" \
  "$deployment#sha256=$launch_sha" "$runtime#sha256=$runtime_sha" \
  "$bridge#sha256=$bridge_sha" "$scratch/c1.json" >/dev/null 2>&1; then
  fail 'c1 release produced a c32 capacity receipt'
fi
write_bind 32 256
cp "$bind" "$deployment/evidence/release-bind.v1"
bind_sha=$(sha256_file "$bind")

tampered=$scratch/tampered.json
sed 's/"authenticated":true/"authenticated":false/' "$receipt" >"$tampered"
tampered_sha=$(sha256_file "$tampered")
if "$verifier" "$bind#sha256=$bind_sha" \
  "$deployment#sha256=$launch_sha" "$runtime#sha256=$runtime_sha" \
  "$bridge#sha256=$bridge_sha" "$tampered#sha256=$tampered_sha" >/dev/null 2>&1; then
  fail 'tampered capacity receipt verified'
fi

foreign_runtime=$scratch/foreign-runtime
printf '%s\n' foreign >"$foreign_runtime"
chmod 0555 "$foreign_runtime"
foreign_sha=$(sha256_file "$foreign_runtime")
if "$verifier" "$bind#sha256=$bind_sha" \
  "$deployment#sha256=$launch_sha" "$foreign_runtime#sha256=$foreign_sha" \
  "$bridge#sha256=$bridge_sha" "$receipt#sha256=$receipt_sha" >/dev/null 2>&1; then
  fail 'runtime substitution verified'
fi

if "$materializer" >/dev/null 2>&1 || "$verifier" >/dev/null 2>&1; then
  fail 'capacity tooling accepted missing arguments'
fi
printf '%s\n' 'Qwen3 authenticated c32 capacity receipt boundary passed'
