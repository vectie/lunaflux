#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

spec_dir="model/spec"
spec_api="$spec_dir/pkg.generated.mbti"
production_spec_files=$(rg --files "$spec_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
independent_spec_encoder='canonical_(paged_)?plan_digest|plan_digest_with_schema|paged_plan_digest_for_valid_batch|lunaflux\.dense-llama-plan'

if ! printf '%s\n' 'pub fn LlamaModelSpec::canonical_plan_digest' | \
  rg -q "$independent_spec_encoder"; then
  echo "independent spec-encoder positive control is ineffective" >&2
  exit 1
fi
if rg -n "$independent_spec_encoder|@crypto|moonbitlang/x/crypto" \
  $production_spec_files "$spec_dir/moon.pkg"; then
  echo "model/spec regained independent plan hashing authority" >&2
  exit 1
fi

metadata_api=$(rg '^pub fn LlamaModelMetadata::' "$spec_api" | sort)
expected_metadata_api=$(printf '%s\n' \
  'pub fn LlamaModelMetadata::content_digest(Self) -> ContentDigest' \
  'pub fn LlamaModelMetadata::from_verified_content(LlamaModelSpec, ContentDigest) -> Self raise ModelSpecError' \
  'pub fn LlamaModelMetadata::spec(Self) -> LlamaModelSpec' | sort)
if [ "$metadata_api" != "$expected_metadata_api" ] || \
  ! rg -U -q 'pub struct LlamaModelMetadata \{[[:space:]]*// private fields[[:space:]]*\}' \
    "$spec_api"; then
  echo "Llama metadata is not the exact opaque spec/content-only API" >&2
  exit 1
fi

spec_identity_api=$(rg '^pub fn (ModelIdentity|PlanDigest)::|^pub fn .* -> .*\b(ModelIdentity|PlanDigest)\b' \
  "$spec_api" | sort)
expected_spec_identity_api=$(printf '%s\n' \
  'pub fn ModelIdentity::content(Self) -> ContentDigest' \
  'pub fn ModelIdentity::new(ContentDigest, PlanDigest) -> Self raise ModelSpecError' \
  'pub fn ModelIdentity::plan(Self) -> PlanDigest' \
  'pub fn PlanDigest::as_hex(Self) -> String' \
  'pub fn PlanDigest::from_sha256(String) -> Self raise ModelSpecError' | sort)
identity_return_pattern='^pub fn (ModelIdentity|PlanDigest)::|^pub fn .* -> .*\b(ModelIdentity|PlanDigest)\b'
if ! printf '%s\n' 'pub fn LlamaModelSpec::forge(Self) -> Array[ModelIdentity]' | \
  rg -q "$identity_return_pattern"; then
  echo "spec identity-return positive control is ineffective" >&2
  exit 1
fi
if [ "$spec_identity_api" != "$expected_spec_identity_api" ]; then
  echo "model/spec exposes a non-generic identity constructor or accessor" >&2
  echo "$spec_identity_api" >&2
  exit 1
fi

mint_pattern='@[A-Za-z_][A-Za-z0-9_]*\.(ModelIdentity::new|PlanDigest::from_sha256)'
identity_claim_files=$(rg -l "$mint_pattern" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
  --glob '!tests/**' . | sort)
expected_identity_claim_files=$(printf '%s\n' \
  './engine/rank_group_wire/stateless_codec.mbt' \
  './engine/tensor_parallel_worker_bootstrap/frame_codec.mbt' \
  './engine/worker_wire/startup_codec.mbt' \
  './kernels/luna_specialization/persisted.mbt' \
  './model/plan/canonical_encoding.mbt' \
  './service/framed_wire/event_codec.mbt' \
  './service/framed_wire/luna_request_view.mbt' | sort)
if ! printf '%s\n' './model/llama/forgery.mbt:@model_spec.ModelIdentity::new' | \
  rg -q "$mint_pattern"; then
  echo "identity claim-site positive control is ineffective" >&2
  exit 1
fi
if [ "$identity_claim_files" != "$expected_identity_claim_files" ]; then
  echo "production identity claim sites escaped the exact mint/decoder allowlist" >&2
  echo "$identity_claim_files" >&2
  exit 1
fi
plan_claim_pattern='@[A-Za-z_][A-Za-z0-9_]*\.PlanDigest::from_sha256'
identity_claim_pattern='@[A-Za-z_][A-Za-z0-9_]*\.ModelIdentity::new'
if [ "$(printf '%s\n' \
  '@spec.PlanDigest::from_sha256(first)' \
  '@model_spec.PlanDigest::from_sha256(second)' | \
  rg -c "$plan_claim_pattern")" -ne 2 ]; then
  echo "identity claim occurrence positive control is ineffective" >&2
  exit 1
fi
for claim_file in \
  engine/rank_group_wire/stateless_codec.mbt \
  engine/tensor_parallel_worker_bootstrap/frame_codec.mbt \
  engine/worker_wire/startup_codec.mbt \
  kernels/luna_specialization/persisted.mbt \
  model/plan/canonical_encoding.mbt \
  service/framed_wire/event_codec.mbt \
  service/framed_wire/luna_request_view.mbt; do
  plan_claim_count=$(rg -c "$plan_claim_pattern" "$claim_file")
  identity_claim_count=$(rg -c "$identity_claim_pattern" "$claim_file")
  if [ "$plan_claim_count" -ne 1 ] || [ "$identity_claim_count" -ne 1 ]; then
    echo "identity mint/decoder occurrence count drifted: $claim_file" >&2
    exit 1
  fi
done

for decoder in \
  engine/rank_group_wire/stateless_codec.mbt \
  engine/tensor_parallel_worker_bootstrap/frame_codec.mbt \
  engine/worker_wire/startup_codec.mbt \
  kernels/luna_specialization/persisted.mbt \
  service/framed_wire/event_codec.mbt \
  service/framed_wire/luna_request_view.mbt; do
  if ! rg -F -q 'Decoded identity is an untrusted claim' "$decoder"; then
    echo "wire/persistence identity decoder lacks its trust label: $decoder" >&2
    exit 1
  fi
done

echo "model/spec identity authority boundary: ok"
