#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

"$root/scripts/validate-mistral-weight-boundary.sh"

moon check \
  deploy/launch_file \
  engine/worker_wire \
  engine/device_step \
  engine/device_worker \
  engine/device_worker_bootstrap \
  runtime/descriptor_file \
  --target native --deny-warn --warn-list +73
moon test deploy/launch_file engine/worker_wire engine/device_step \
  engine/device_worker engine/device_worker_bootstrap runtime/descriptor_file \
  --target native --deny-warn --warn-list +73
moon test ops/runtime_instance cmd/lunaflux --target native

if rg -ni 'mistral' scheduler kv service engine/device_step engine/device_worker \
  --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  echo "Mistral introduced a family branch into a shared runtime owner" >&2
  exit 1
fi

for anchor in \
  'DenseMistralBf16PagedAotV7' \
  'admit_mistral_bootstrap_source_v7' \
  'decode_mistral_bootstrap_source_v7'
do
  if ! rg -F "$anchor" engine/worker_wire deploy/launch_file >/dev/null; then
    echo "Mistral route lost exact source/launch anchor: $anchor" >&2
    exit 1
  fi
done

for anchor in \
  'lunaflux.runtime.mistral_bf16.v1' \
  'load_mistral_v1' \
  'load_mistral_v1_materialized' \
  'route_manifest_sha256()' \
  'sliding_window_tokens'
do
  if ! rg -F "$anchor" runtime/descriptor_file >/dev/null; then
    echo "Mistral descriptor lost exact identity anchor: $anchor" >&2
    exit 1
  fi
done

for anchor in \
  'prepare_numeric_bf16_paged_graph_executor_v1' \
  'admit_numeric_bf16_plan_v1'
do
  if ! rg -F "$anchor" engine/device_step engine/device_worker \
    engine/device_worker_bootstrap >/dev/null; then
    echo "Mistral route lost shared numeric BF16 owner: $anchor" >&2
    exit 1
  fi
done

if ! rg -n 'DenseMistralBf16PagedAotV7 => mistral_loader\(\)' \
  ops/runtime_instance/runtime_route.mbt >/dev/null; then
  echo "authenticated launch recipe no longer selects exactly one Mistral loader" >&2
  exit 1
fi

for anchor in \
  'Mistral v7 full frame roundtrips exact authenticated claims' \
  'Mistral v7 rejects malformed reserved integrity and framing bytes' \
  'selected Mistral loader failure never probes legacy or I8 fallback' \
  'opaque run CLI admits exactly one deployment argument'
do
  if ! rg -F "$anchor" engine/worker_wire ops/runtime_instance cmd/lunaflux \
    --glob '*test.mbt' >/dev/null; then
    echo "Mistral route lost hostile lifecycle coverage: $anchor" >&2
    exit 1
  fi
done

if ! rg -n 'resources_complete\(\).*readiness|readiness_contract' \
  engine/device_worker --glob '*.mbt' >/dev/null; then
  echo "shared worker readiness completion gate drifted" >&2
  exit 1
fi

if rg -ni '(physical validation: pass|physically validated|readiness: true)' \
  engine/device_worker_bootstrap runtime/descriptor_file ops/runtime_instance \
  --glob '*.mbt' >/dev/null; then
  echo "Mistral route fabricated readiness or physical evidence" >&2
  exit 1
fi

for file in \
  engine/device_step/numeric_bf16_executor.mbt \
  engine/device_worker_bootstrap/derive_mistral.mbt \
  engine/device_worker_bootstrap/prepare_mistral.mbt \
  engine/worker_wire/bootstrap_source_mistral_codec.mbt \
  runtime/descriptor_file/mistral_load.mbt \
  runtime/descriptor_file/mistral_schema.mbt
do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "Mistral runtime route file exceeds size boundary: $file ($lines)" >&2
    exit 1
  fi
done

echo "Mistral shared BF16 runtime route boundary: ok"
