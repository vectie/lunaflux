#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

validator="$repo_root/scripts/validate-authenticated-device-materialization-boundary.sh"
temporary=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-auth-materialize.XXXXXX")
trap 'rm -rf "$temporary"' EXIT HUP INT TERM

sh "$validator"

cp model/device_materialize/pkg.generated.mbti "$temporary/pkg.generated.mbti"
printf '%s\n' \
  'pub fn load(@device.Context, @llama_weights.LlamaWeightBindings, Bytes) -> DeviceWeightPreparation' \
  >>"$temporary/pkg.generated.mbti"
if LUNA_DEVICE_MATERIALIZE_API="$temporary/pkg.generated.mbti" \
  sh "$validator" >"$temporary/raw-api.out" 2>"$temporary/raw-api.err"; then
  echo "authenticated device materialization hostile boundary: raw API escaped" >&2
  exit 1
fi
rg -F -q 'generated API exposes raw bytes as device-weight authority' \
  "$temporary/raw-api.err"

cp model/device_materialize/load.mbt "$temporary/load.mbt"
printf '%s\n' \
  'pub fn load_unchecked(file : BytesView) -> DeviceShardedWeightPreparation' \
  >>"$temporary/load.mbt"
if LUNA_DEVICE_MATERIALIZE_SOURCE="$temporary/load.mbt" \
  sh "$validator" >"$temporary/raw-source.out" 2>"$temporary/raw-source.err"; then
  echo "authenticated device materialization hostile boundary: raw source escaped" >&2
  exit 1
fi
rg -F -q 'raw in-memory device loader is public' "$temporary/raw-source.err"

echo "authenticated device materialization hostile boundary: pass"
