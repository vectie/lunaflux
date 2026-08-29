#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

source_file=${LUNA_DEVICE_MATERIALIZE_SOURCE:-model/device_materialize/load.mbt}
api_file=${LUNA_DEVICE_MATERIALIZE_API:-model/device_materialize/pkg.generated.mbti}

fail() {
  echo "authenticated device materialization boundary: $1" >&2
  exit 1
}

[ -f "$source_file" ] || fail "missing loader source: $source_file"
[ -f "$api_file" ] || fail "missing generated API: $api_file"

raw_loader_pattern='^pub fn [A-Za-z0-9_]*load[A-Za-z0-9_]*\([^\n]*(Bytes|BytesView)[^\n]*\) -> Device(Sharded)?WeightPreparation'

for positive_control in \
  'pub fn load(@device.Context, @llama_weights.LlamaWeightBindings, Bytes) -> DeviceWeightPreparation' \
  'pub fn load_sharded(BytesView) -> DeviceShardedWeightPreparation'; do
  if ! printf '%s\n' "$positive_control" | rg -q "$raw_loader_pattern"; then
    fail "raw-loader positive control is ineffective: $positive_control"
  fi
done

if rg -U -n "$raw_loader_pattern" "$source_file"; then
  fail "raw in-memory device loader is public"
fi
if rg -n "$raw_loader_pattern" "$api_file"; then
  fail "generated API exposes raw bytes as device-weight authority"
fi

if rg -n '@materialize\.|vectie/lunaflux/model/materialize|MaterializationFailed' \
  model/device_materialize/moon.pkg model/device_materialize \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "device materializer retained unchecked in-memory materialization authority"
fi

for required in \
  'pub fn inspect_file(@approved_fs.ApprovedRoot, @spec.LlamaModelMetadata, @plan.ModelPlan, @approved_fs.ApprovedRelativeLocator,' \
  'pub fn load_file(@device.Context, @approved_fs.ApprovedRoot, @spec.LlamaModelMetadata, @plan.ModelPlan, @approved_fs.ApprovedRelativeLocator,' \
  'pub fn load_inspected_file(@device.Context, @approved_fs.ApprovedRoot, DeviceWeightFileInspection) -> DeviceWeightPreparation'; do
  rg -F -q "$required" "$api_file" || \
    fail "authenticated file API changed or disappeared: $required"
done

visitor_owners=$(rg -l '@materialize\.visit\(' model/device_materialize \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
if [ -n "$visitor_owners" ]; then
  fail "unchecked in-memory visitor acquired device authority: $visitor_owners"
fi

echo "authenticated device materialization boundary: pass"
