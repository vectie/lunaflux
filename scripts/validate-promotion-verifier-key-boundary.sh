#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if [ "$(rg -c 'extern\s+"[cC]"' internal/promotion_verifier_key/ffi.mbt)" -ne 6 ] ||
  rg -n 'extern\s+"[cC]"|#external' internal/promotion_verifier_key \
    --glob '*.mbt' --glob '!ffi.mbt'; then
  printf '%s\n' 'promotion verifier declarations drifted from one exact FFI file' >&2
  exit 1
fi

for required in \
  'LF_PROMOTION_KEY_FIXED_FD 7' \
  'AF_UNIX' \
  'SOCK_STREAM' \
  'F_DUPFD_CLOEXEC' \
  'O_NONBLOCK' \
  'MSG_PEEK' \
  'lunaflux_promotion_verifier_key_close'; do
  if ! rg -q "$required" internal/promotion_verifier_key/promotion_verifier_key.c; then
    printf 'promotion verifier ABI lost required anchor: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'bind\s*\(|listen\s*\(|accept\s*\(|connect\s*\(|getenv\s*\(|system\s*\(|popen\s*\(' \
  internal/promotion_verifier_key kernels/luna_capability_manifest/startup_verifier_key.mbt; then
  printf '%s\n' 'promotion verifier gained ambient, network, or execution authority' >&2
  exit 1
fi

if rg -n 'pub fn LunaExternalApprovalVerifier::new|pub fn .*from_(key|bytes)|pub fn .*::(key|key_bytes|raw_key)\(' \
  kernels/luna_capability_manifest --glob '*.mbt'; then
  printf '%s\n' 'promotion verifier exposed an arbitrary-key or key-observation API' >&2
  exit 1
fi

if ! rg -q 'vectie/lunaflux/internal/promotion_verifier_key' \
    kernels/luna_capability_manifest/moon.pkg ||
  rg -n 'internal/promotion_verifier_key' --glob 'moon.pkg' \
    --glob '!kernels/luna_capability_manifest/moon.pkg' \
    --glob '!internal/promotion_verifier_key/moon.pkg'; then
  printf '%s\n' 'only the Luna capability owner may import the verifier-key ABI' >&2
  exit 1
fi

if rg -n 'promotion_verifier_key|LunaStartupExternalApprovalHandoff|authenticate_manifest_claim' \
  scheduler service sampling engine/device_step engine/device_worker \
  --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'startup promotion authentication escaped into request/token execution' >&2
  exit 1
fi

private_parent_constructor_calls=$(rg -l \
  'LunaParentStartupApprovalWitness::from_private_parent_channel_v1' \
  --glob '*.mbt' \
  --glob '!kernels/luna_capability_manifest/parent_startup_attestation.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
if [ "$private_parent_constructor_calls" != \
    'engine/worker_wire/parent_approval_attestation.mbt' ]; then
  printf '%s\n' \
    'parent approval witness construction escaped its one private-channel codec' >&2
  printf '%s\n' "$private_parent_constructor_calls" >&2
  exit 1
fi

parent_attestation_decoder_calls=$(rg -l \
  'decode_parent_approval_attestation_v1' \
  --glob '*.mbt' \
  --glob '!engine/worker_wire/parent_approval_attestation.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
if [ "$parent_attestation_decoder_calls" != \
    'engine/device_worker_child/run.mbt' ]; then
  printf '%s\n' \
    'parent approval attestation decode escaped the inherited child channel' >&2
  printf '%s\n' "$parent_attestation_decoder_calls" >&2
  exit 1
fi

launch_identity_calls=$(rg -l '\.require_launch_identity\(' \
  --glob '*.mbt' \
  --glob '!kernels/luna_capability_manifest/parent_startup_attestation.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
if [ "$launch_identity_calls" != 'engine/device_worker_bootstrap/prepare.mbt' ]; then
  printf '%s\n' \
    'parent approval launch identity validation escaped child bootstrap' >&2
  printf '%s\n' "$launch_identity_calls" >&2
  exit 1
fi

if rg -n \
  'LunaParentStartupApprovalWitness|parent_approval_attestation|parent_approval' \
  scheduler service sampling engine/device_step engine/device_worker \
  --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'parent approval authority escaped into request/token execution' >&2
  exit 1
fi

if rg -n 'pub fn encode_parent_approval_attestation_v1' \
    engine/worker_wire/parent_approval_attestation.mbt; then
  printf '%s\n' 'parent attestation exposed a raw-subject public encoder' >&2
  exit 1
fi

promotion_line=$(rg -n -m 1 'let promotion = acquire_luna_promotion_verifier\(\)' \
  cmd/lunaflux/native_run.mbt | cut -d: -f1)
drain_line=$(rg -n -m 1 'let drain = @inherited_drain.prepare_inherited_drain_v1\(\)' \
  cmd/lunaflux/native_run.mbt | cut -d: -f1)
owner_line=$(rg -n -m 1 'let owner = @runtime_instance\.prepare_opaque_cli\(' \
  cmd/lunaflux/native_run.mbt | cut -d: -f1)
if [ -z "$promotion_line" ] || [ -z "$drain_line" ] || [ -z "$owner_line" ] ||
  [ "$promotion_line" -ge "$drain_line" ] || [ "$drain_line" -ge "$owner_line" ]; then
  printf '%s\n' 'promotion verifier startup acquisition ordering drifted' >&2
  exit 1
fi

moon info >/dev/null
if rg -n 'internal/promotion_verifier_key|NativePromotionVerifierKey|PromotionVerifierKeyHandle' \
  kernels/luna_capability_manifest/pkg.generated.mbti \
  ops/runtime_instance/pkg.generated.mbti; then
  printf '%s\n' 'internal promotion verifier authority leaked publicly' >&2
  exit 1
fi

for source_file in internal/promotion_verifier_key/*.mbt \
  internal/promotion_verifier_key/*.c \
  kernels/luna_capability_manifest/startup_verifier_key.mbt \
  kernels/luna_capability_manifest/parent_startup_attestation.mbt \
  engine/worker_wire/parent_approval_attestation.mbt; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; verifier-key files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux startup promotion-verifier authority boundary is valid.'
