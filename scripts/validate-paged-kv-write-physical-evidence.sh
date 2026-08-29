#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
package=release/luna_paged_kv_write_physical_evidence
fail() { printf 'paged KV-write evidence boundary failed: %s\n' "$1" >&2; exit 1; }

for file in "$package"/*.mbt; do
  lines=$(wc -l <"$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "file reached 500 lines: $file"
done

for dependency in \
  'vectie/lunaflux/internal/canonical_sha256' \
  'vectie/lunaflux/kernels/catalog' \
  'vectie/lunaflux/kernels/launch_contract' \
  'vectie/lunaflux/kernels/luna_cuda_paged_kv_write_aot' \
  'vectie/lunaflux/model/plan'; do
  grep -Fq "\"$dependency\"" "$package/moon.pkg" || fail "typed dependency missing: $dependency"
done

if rg -n 'moonbitlang/async|internal/(cuda|process|approved_fs)|vectie/lunaflux/(device|engine|runtime|scheduler|service|cmd)/' "$package/moon.pkg" >/dev/null; then
  fail 'evidence package gained active runtime authority'
fi
if rg -n 'extern[[:space:]]+"c"|@fs\.|@process\.|@cuda\.|async[[:space:]]+fn|pub fn (open|spawn|execute|promote|bind_manifest)\(' "$package" --glob '*.mbt' --glob '!**/*_wbtest.mbt' >/dev/null; then
  fail 'evidence source gained active authority'
fi

for anchor in \
  'lunaflux-paged-kv-write-physical-evidence.v1' \
  'serial_oracle_source_sha256' \
  'scalar_referee_source_sha256' \
  'probe_executable_sha256' \
  'dispatch_canary_exact' \
  'context_device_uuid' \
  'context_device_pci' \
  'sass_global_load_observed", "true"' \
  'sass_global_store_observed", "true"' \
  'physical_cuda_observed", "true"' \
  'synthetic_test_only", "false"' \
  'reader.finish()' \
  'manifest_bindable", "false"' \
  'promotion_authority", "absent"' \
  'runtime_dispatch_authority", "absent"'; do
  rg -Fq "$anchor" "$package" || fail "canonical admission anchor missing: $anchor"
done

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null
interface=$package/pkg.generated.mbti
grep -Fqx 'pub fn admit_qualified_paged_kv_write_artifact(@luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate, @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact, Bytes, expected_evidence_digest~ : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest, expected_approved_policy_digest~ : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest) -> QualifiedPagedKvWriteArtifact raise PagedKvWritePhysicalError' "$interface" || fail 'admission signature drifted'
rg -U -q 'pub struct QualifiedPagedKvWriteArtifact \{\n  // private fields\n\}' "$interface" || fail 'qualified wrapper is not opaque'
if rg -n '^pub fn QualifiedPagedKvWriteArtifact::(new|make|create|open|load|execute|promote|bind_manifest)' "$interface" >/dev/null; then
  fail 'qualified wrapper gained construction or runtime authority'
fi
for accessor in candidate_digest source_digest recipe_digest compiled_binding_digest module_digest compile_receipt_digest evidence_digest files_manifest_digest outer_seal_digest device_uuid device_pci positioned_qkv_activation dispatch_canary_per_token fallback_source_digest_sha256 fallback_recipe_digest_sha256; do
  rg -Fq "pub fn QualifiedPagedKvWriteArtifact::$accessor" "$interface" || fail "opaque prerequisite accessor missing: $accessor"
done

printf '%s\n' 'Paged KV-write physical evidence remains opaque, qualification-only, and positionally bound.'
