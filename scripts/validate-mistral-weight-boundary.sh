#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package="$root/model/mistral_weights"

"$root/scripts/validate-mistral-family-boundary.sh"

moon check model/mistral_spec model/mistral model/mistral_weights \
  --target native --deny-warn --warn-list +73
moon test model/mistral_weights \
  --target native --deny-warn --warn-list +73

if rg -n 'vectie/lunaflux/(device("|/)|scheduler|kv|prefix|api|service|engine|kernels|internal/cuda)' \
  "$package/moon.pkg" >/dev/null; then
  echo "Mistral weight facade crosses a runtime or native-ABI boundary" >&2
  exit 1
fi

for dependency in \
  'vectie/lunaflux/model/device_materialize' \
  'vectie/lunaflux/model/mistral' \
  'vectie/lunaflux/model/numeric_weight_file' \
  'vectie/lunaflux/runtime/approved_fs'
do
  if ! rg -F "$dependency" "$package/moon.pkg" >/dev/null; then
    echo "Mistral weight facade lost required generic owner: $dependency" >&2
    exit 1
  fi
done

for tensor_name in \
  'model.embed_tokens.weight' \
  'input_layernorm.weight' \
  'self_attn.q_proj.weight' \
  'self_attn.k_proj.weight' \
  'self_attn.v_proj.weight' \
  'self_attn.o_proj.weight' \
  'post_attention_layernorm.weight' \
  'mlp.gate_proj.weight' \
  'mlp.up_proj.weight' \
  'mlp.down_proj.weight' \
  'model.norm.weight' \
  'lm_head.weight'
do
  if ! rg -F "$tensor_name" "$package/bind.mbt" >/dev/null; then
    echo "Mistral tensor translation is incomplete: $tensor_name" >&2
    exit 1
  fi
done

if ! rg -n '@mistral\.build\(metadata\)|@mistral\.build_paged\(metadata' \
  "$package/bind.mbt" "$package/file.mbt" >/dev/null; then
  echo "Mistral weight admission no longer rebuilds the family plan" >&2
  exit 1
fi

if ! rg -n '@numeric_weight_file\.inspect_file' \
  "$package/file.mbt" >/dev/null; then
  echo "Mistral facade no longer delegates admission to the generic numeric owner" >&2
  exit 1
fi

if ! rg -n 'MistralWeightFileInspection::generic_inspection' \
  "$package/types.mbt" >/dev/null; then
  echo "Mistral facade no longer projects shared numeric inspection evidence" >&2
  exit 1
fi

if rg -n '@numeric_weight_file\.load_inspected_file|^pub fn load_inspected_file' \
  "$package" --glob '*.mbt' --glob 'pkg.generated.mbti' >/dev/null; then
  echo "Mistral facade duplicated the generic device materialization route" >&2
  exit 1
fi

if rg -n '(@device\.Context::|@[A-Za-z0-9_]*executor|load_module|launch_kernel|Readiness)' \
  "$package" --glob '*.mbt' >/dev/null; then
  echo "Mistral weight package fabricated runtime or execution authority" >&2
  exit 1
fi

for hostile_anchor in \
  'missing Mistral tensor was admitted' \
  'extra Mistral tensor was admitted' \
  'mis-shaped Mistral tensor was admitted' \
  'sliding-window replay' \
  'foreign artifact digest' \
  'root.close()'
do
  if ! rg -F "$hostile_anchor" "$package" --glob '*test.mbt' >/dev/null; then
    echo "Mistral hostile coverage is missing: $hostile_anchor" >&2
    exit 1
  fi
done

fixture="$package/fixtures/tiny_mistral_v1.json"
for fixture_anchor in \
  '"license": "CC0-1.0"' \
  '"redistributable": true' \
  '"synthetic": true' \
  '"operations": 21' \
  '"tensors": 21'
do
  if ! rg -F "$fixture_anchor" "$fixture" >/dev/null; then
    echo "Mistral synthetic corpus is not pinned honestly: $fixture_anchor" >&2
    exit 1
  fi
done

for file in "$package"/*.mbt "$package"/*.md
do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "Mistral weight file exceeds the package size limit: $file ($lines)" >&2
    exit 1
  fi
done

echo "Mistral weight/materialization boundary: ok"
