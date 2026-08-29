#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=kernels/luna_cuda_paged_attention_readonly_aot
exporter=tests/paged_attention_readonly_source_export

moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null
moon check "$exporter" --target native --deny-warn --warn-list +73
moon build "$exporter" --target native --deny-warn --warn-list +73
bash -n scripts/run-paged-attention-readonly-source-campaign.sh
bash -n scripts/test-paged-attention-readonly-source-campaign.sh

for required in \
  'family=paged-attention-rotated-q-paged-kv-readonly' \
  'selection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1' \
  'rotated_query_only=true' \
  'kv_layout=layer_major_split_key_value_v1' \
  'kv_layout_stable_version=1' \
  'kv_cache_access=read_only' \
  'kv_cache_mutation=none' \
  'full_qkv_row_stride_bound=true' \
  'dispatch_canary_publication=exactly-once-after-output' \
  'fallback_required=true' \
  'manifest_bindable=false' \
  'promotion_authority=none'; do
  rg -F -q "$required" "$package" --glob '*.mbt' || {
    printf 'paged-attention read-only boundary missing: %s\n' "$required" >&2
    exit 1
  }
done

rg -F -q 'readonly_cuda_source(symbol, reference, input_row_width, None)' \
  "$package/source_production.mbt"
rg -F -q 'diagnostic_canary=absent' "$package/lower.mbt"
if rg -F 'Some(' "$package/source_production.mbt"; then
  printf '%s\n' 'production readonly source path enabled diagnostics' >&2
  exit 1
fi

if rg -n 'RequireExternallyQualified|manifest_bindable=true|promotion_authority=(present|granted)|runtime.?dispatch|internal/cuda|nvrtc|NVRTC' \
  "$package" "$exporter" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'paged-attention read-only candidate crossed runtime authority' >&2
  exit 1
fi
if rg -n '(key_cache|value_cache)\[[^]]+\][[:space:]]*=' "$package/source.mbt"; then
  printf '%s\n' 'paged-attention read-only source contains KV mutation' >&2
  exit 1
fi
for file in $(rg --files "$package" "$exporter" -g '*.mbt'); do
  lines=$(wc -l <"$file")
  [ "$lines" -lt 500 ] || {
    printf 'paged-attention read-only MoonBit file exceeds debt budget: %s (%s)\n' "$file" "$lines" >&2
    exit 1
  }
done
scripts/test-paged-attention-readonly-source-campaign.sh
scripts/validate-paged-attention-readonly-physical-evidence.sh
printf '%s\n' 'paged-attention read-only AOT boundary passed'
