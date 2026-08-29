#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "sha256sum or shasum is required" >&2
    exit 1
  fi
}

fixture=tests/mistral_reference_corpus
binding=$fixture/binding.json
config=$fixture/mistral_config.json
weights=tests/reference_corpus/model.safetensors
corpus=tests/reference_corpus/corpus.json

[ "$(sha256_file "$binding")" = \
  957f162b471e1451943b809374ec5c5d833fbc3be86a238b7d81c84a509a0b2d ] || {
  echo "Mistral correctness binding identity changed" >&2
  exit 1
}
[ "$(sha256_file "$config")" = \
  c1ab500fc14070e8ad822e5b5b896d5be80cfbc6f4c7d947064ef0d0a302f866 ] || {
  echo "Mistral semantic configuration identity changed" >&2
  exit 1
}
[ "$(sha256_file "$weights")" = \
  852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30 ] || {
  echo "approved BF16 reference weights changed" >&2
  exit 1
}
[ "$(sha256_file "$corpus")" = \
  f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0 ] || {
  echo "independently generated reference corpus changed" >&2
  exit 1
}

for honesty_anchor in \
  '"source_model_family": "LlamaForCausalLM"' \
  '"admitted_model_family": "MistralForCausalLM"' \
  '"fixture_only": true' \
  '"production_mistral_artifact": false' \
  '"physical_execution_evidence": false' \
  '"semantics": "exact-dense-plan-equivalence-at-or-below-sliding-window"'
do
  if ! rg -F "$honesty_anchor" "$binding" >/dev/null; then
    echo "Mistral correctness corpus lost honesty anchor: $honesty_anchor" >&2
    exit 1
  fi
done

for correctness_anchor in \
  'assert_eq(mistral_plan, llama.plan())' \
  '@mistral_weights.bind' \
  'assert_mistral_logit' \
  'execution.selection().token_id()' \
  'max_new_tokens=4' \
  'assert_not_eq(sha256_hex(weights[1:]), weights_digest)' \
  'assert_not_eq(@mistral.build(replay_metadata), llama.plan())' \
  'assert_not_eq(sha256_hex(hostile_config), config_digest)' \
  'assert_not_eq(sha256_hex(hostile_corpus), corpus_digest)'
do
  if ! rg -F "$correctness_anchor" \
    "$fixture/mistral_reference_corpus_test.mbt" >/dev/null; then
    echo "Mistral correctness corpus lost gate: $correctness_anchor" >&2
    exit 1
  fi
done

if rg -ni \
  '(physical validation: pass|physically validated|sm89[^\n]*pass|sm90[^\n]*pass|production_mistral_artifact"[[:space:]]*:[[:space:]]*true)' \
  "$fixture" --glob '*.mbt' --glob '*.json' >/dev/null; then
  echo "Mistral correctness corpus fabricated physical or production authority" >&2
  exit 1
fi

for file in "$fixture"/*.mbt "$fixture"/*.json
do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "Mistral correctness fixture exceeds size boundary: $file ($lines)" >&2
    exit 1
  fi
done

"$root/scripts/validate-mistral-weight-boundary.sh"
moon check tests/mistral_reference_corpus \
  --target native --deny-warn --warn-list +73
moon test tests/mistral_reference_corpus \
  --target native --deny-warn --warn-list +73

echo "Mistral authenticated semantic correctness corpus: ok"
