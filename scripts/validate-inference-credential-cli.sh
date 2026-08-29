#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

source_file=cmd/lunaflux/native_run.mbt

line_of() {
  pattern=$1
  rg -n -m 1 "$pattern" "$source_file" | cut -d: -f1
}

credential_line=$(line_of 'let credential = acquire_opaque_inference_credential()')
drain_line=$(line_of 'let drain = @inherited_drain.prepare_inherited_drain_v1()')
owner_line=$(line_of 'let owner = @runtime_instance\.prepare_opaque_cli\(')
attach_line=$(line_of 'owner\.attach_inherited_drain_v1\(drain\)')

if [ -z "$credential_line" ] || [ -z "$drain_line" ] || \
  [ -z "$owner_line" ] || [ -z "$attach_line" ] || \
  [ "$credential_line" -ge "$drain_line" ] || \
  [ "$drain_line" -ge "$owner_line" ] || \
  [ "$owner_line" -ge "$attach_line" ]; then
  printf '%s\n' 'opaque CLI capability acquisition/activation ordering drifted' >&2
  exit 1
fi

if rg -n 'getenv|@env|open_absolute|ApprovedRoot|path|locator|sha256' \
  runtime/inference_credential internal/inference_credential \
  --glob '*.mbt' --glob '*.c' --glob '*.h'; then
  printf '%s\n' 'credential capability acquired ambient or filesystem authority' >&2
  exit 1
fi

if rg -n 'pub fn .*credential_(bytes|length|digest)|println\([^)]*credential|impl (Debug|Show) for InheritedInference' \
  runtime/inference_credential ops/runtime_instance cmd/lunaflux \
  --glob '*.mbt'; then
  printf '%s\n' 'credential path gained an observable secret representation' >&2
  exit 1
fi

for required in \
  'pub fn InheritedInferenceAuthPolicy::close' \
  'inherited\.close\(\)' \
  'startup_binding\.close_authentication\(\)' \
  'self\.auth_policy\.close\(\)'; do
  if ! rg -q "$required" runtime/inference_credential ops/runtime_instance \
    service/online_tcp --glob '*.mbt'; then
    printf 'credential zeroize lifecycle is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if ! rg -U -q \
  'self\.http_workspace\.close_authentication\(\)\n    self\.auth_policy\.close\(\)\n    self\.phase = LUNA_ONLINE_TCP_SERVER_CLOSED' \
  service/online_tcp/openai_server_progress.mbt ||
  ! rg -U -q \
  'slot\.http_workspace\.close_authentication\(\)\n    \}\n    self\.auth_policy\.close\(\)\n    self\.closed = true' \
  service/online_tcp/openai_pool_progress.mbt; then
  printf '%s\n' 'OpenAI terminal auth-policy wipe ordering drifted' >&2
  exit 1
fi

moon build cmd/lunaflux --target native --release --deny-warn

printf '%s\n' 'LunaFlux opaque CLI credential activation boundary is valid.'
