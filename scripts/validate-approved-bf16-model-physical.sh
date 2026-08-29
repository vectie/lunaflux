#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
probe_dir=tests/approved_bf16_model_physical

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

check_hash() {
  expected=$1
  path=$2
  actual=$(sha256_file "$path")
  [ "$actual" = "$expected" ] || {
    echo "approved-model fixture digest mismatch: $path" >&2
    exit 1
  }
}

check_hash 9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622 \
  tests/reference_corpus/config.json
check_hash 852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30 \
  tests/reference_corpus/model.safetensors
check_hash fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b \
  tests/reference_corpus/upstream_tokenizer.json
check_hash f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0 \
  tests/reference_corpus/corpus.json

for anchor in \
  'ExecutedPagedGraphExecution' \
  'token_ids.length() > 32' \
  'copy_to_fixed_host' \
  'scratch.encoded_logits' \
  'value.is_nan()' \
  'value.is_inf()'; do
  rg -Fq "$anchor" engine/device_step/paged_executor_observe.mbt || {
    echo "selected-logit boundary anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  '0, 1, 2, 42, 99, 333, 897, 1031, 2049, 2185, 2999' \
  'APPROVED_SELECTED_LOGIT_TOLERANCE : Double = 0.01' \
  'continuation != [1031, 2185, 688, 2844]' \
  'APPROVED_TOKENS_PER_PAGE : Int = 8' \
  'APPROVED_CONTEXT_PAGE_COUNT : Int = 32' \
  'APPROVED_CONTEXT_TOKENS : Int = 256' \
  'graph_policy=PagedCapturedRequired'; do
  rg -Fq "$anchor" "$probe_dir" || {
    echo "approved-model numerical anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'expected_toolchain_sha256' \
  'expected_driver_identity_sha256' \
  'module_relative_path' \
  'first_sha != module_sha' \
  'second_sha != module_sha'; do
  rg -Fq "$anchor" "$probe_dir/compiled_set.mbt" || {
    echo "compiled-set authentication anchor is missing: $anchor" >&2
    exit 1
  }
done

uses=$(rg -l 'observe_offline_selected_logits' --glob '*.mbt' . |
  sed 's#^\./##' | LC_ALL=C sort)
allowed='engine/device_step/paged_executor_observe.mbt
engine/device_step/paged_executor_observe_wbtest.mbt
tests/approved_bf16_model_physical/schedule.mbt'
[ "$uses" = "$allowed" ] || {
  echo 'offline selected-logit observer escaped its bounded validation scope' >&2
  printf '%s\n' "$uses" >&2
  exit 1
}

if rg -n 'observe_offline_selected_logits' scheduler service cmd \
  --glob '*.mbt' >/dev/null 2>&1; then
  echo 'offline selected-logit observer leaked into scheduler/serving code' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/internal/cuda|extern\s+"[cC]"|#external' \
  "$probe_dir" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  echo 'approved-model probe bypasses the public device boundary' >&2
  exit 1
fi

if rg -n 'nvrtc|--ptx|output=ptx|code=compute_[0-9]+[^,)]' \
  "$probe_dir" scripts/probe-approved-bf16-model-cuda.sh \
  --glob '*.mbt' --glob '*.sh' >/dev/null 2>&1; then
  echo 'approved-model probe introduced runtime compilation or PTX' >&2
  exit 1
fi

for anchor in \
  'scripts/verify-luna-bf16-kernel-set.sh "$compiled_root"' \
  'cp -R "$compiled_root/sha256" "$runtime_root/sha256"' \
  'runtime.stderr' \
  'scope=tiny-approved-model-numerics'; do
  rg -Fq "$anchor" scripts/probe-approved-bf16-model-cuda.sh || {
    echo "physical runner boundary anchor is missing: $anchor" >&2
    exit 1
  }
done

for source in "$probe_dir"/*.mbt engine/device_step/paged_executor_observe.mbt; do
  lines=$(wc -l <"$source" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    echo "approved-model source exceeds 500 lines: $source ($lines)" >&2
    exit 1
  }
done

sh -n scripts/probe-approved-bf16-model-cuda.sh
moon check --target native --deny-warn "$probe_dir"
moon test --target native --deny-warn "$probe_dir"
moon check --target native --deny-warn engine/device_step
echo 'LunaFlux approved tiny BF16 model physical boundary passed.'
