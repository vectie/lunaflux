#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

device_source="device/numeric_capability.mbt"
kernel_dir="kernels/numeric_capability_manifest"
engine_dir="engine/fp8_startup_admission"

fail() {
  echo "FP8 startup admission boundary: $1" >&2
  exit 1
}

for path in "$device_source" "$kernel_dir" "$engine_dir"; do
  [ -e "$path" ] || fail "missing $path"
done

if rg -n 'vectie/lunaflux/(model|kernels|engine|scheduler|kv)/' device/moon.pkg; then
  fail "device capability acquired model, kernel, engine, scheduler, or KV authority"
fi

if rg -n 'vectie/lunaflux/(scheduler|kv|engine|internal/(cuda|nccl))' \
  "$kernel_dir/moon.pkg" "$engine_dir/moon.pkg"; then
  fail "FP8 admission acquired scheduler, KV, engine-cycle, or native ABI authority"
fi

if rg -n 'extern "c"|open_context|load_module|create_stream|\.run\(|\.execute\(' \
  "$device_source" "$kernel_dir" "$engine_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "inert FP8 admission acquired physical execution authority"
fi

for required in \
  'FiniteFp8E4M3W8A8F32AccumulateBf16OutputV1' \
  'lunaflux.device-numeric-capability.v1\x00'; do
  rg -F -q "$required" "$device_source" || \
    fail "closed device capability invariant is missing: $required"
done

fp8_policy=$(sed -n \
  '/^fn fp8_e4m3_architecture_allowed_v1(/,/^}/p' "$device_source")
for required in \
  '(major == 8 && minor == 9)' \
  '(major == 9 && minor == 0)' \
  '(major == 12 && minor == 0)'; do
  printf '%s\n' "$fp8_policy" | rg -F -q "$required" || \
    fail "closed FP8 target is missing: $required"
done

if rg -n 'major[[:space:]]*>|minor[[:space:]]*>=' "$device_source"; then
  fail "device capability silently admits future architectures"
fi

for required in \
  'activation_compute() ==' \
  'ActivationScalePolicy::dynamic_per_tensor_f32_v1' \
  'TensorStorageDigest' \
  'ScaleTensorRef' \
  'VendorOrBf16Fallback' \
  'pub struct AdmittedFp8KernelArtifacts' \
  'priv artifacts : @artifact.KernelArtifactBundle' \
  'lunaflux.fp8-kernel-capability-manifest.v1\x00'; do
  rg -F -q "$required" "$kernel_dir" --glob '*.mbt' || \
    fail "kernel manifest invariant is missing: $required"
done

artifact_owner_count=$(rg -F -c \
  'priv artifacts : @artifact.KernelArtifactBundle' \
  "$kernel_dir"/*.mbt | awk -F: '{ total += $2 } END { print total + 0 }')
[ "$artifact_owner_count" -eq 1 ] || \
  fail "kernel manifest must retain exactly one aggregate artifact bundle"

for required in \
  'pub struct Fp8StartupAdmission' \
  'manifest.numeric_schema_digest() != plan.numeric_binding().digest()' \
  'manifest.device_numeric_digest() != device_numeric.digest()' \
  'tensor.storage().digest() == weight.storage_digest()' \
  'lunaflux.fp8-startup-admission.v1\x00'; do
  rg -F -q "$required" "$engine_dir" --glob '*.mbt' || \
    fail "engine startup invariant is missing: $required"
done

if rg -n '[A-Za-z0-9_]+\?[[:space:]]*:' \
  "$device_source" "$kernel_dir" "$engine_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "production FP8 admission introduced an optional compatibility field"
fi

for directory in device "$kernel_dir" "$engine_dir"; do
  for file in "$directory"/*.mbt; do
    case "$file" in
      *_test.mbt|*_wbtest.mbt) continue ;;
    esac
    lines=$(wc -l < "$file" | tr -d ' ')
    [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line production budget"
  done
done

echo "FP8 device, kernel, and inert startup admission boundaries: ok"
