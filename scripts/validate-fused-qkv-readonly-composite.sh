#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

moon check release/luna_paged_attention_readonly_production --target native --deny-warn
moon test release/luna_paged_attention_readonly_production --target native --deny-warn
moon check engine/device_step --target native --deny-warn
moon test engine/device_step --target native --deny-warn

rg -q 'dispatch_authority=false' engine/device_step/fused_qkv_readonly_admit.mbt
rg -q 'single-stream-writer-before-reader' engine/device_step/fused_qkv_readonly_admit.mbt
rg -q 'atomic-three-to-two' engine/device_step/fused_qkv_readonly_admit.mbt
rg -q 'production_artifacts=absent' engine/device_step/fused_qkv_readonly_admit.mbt
rg -q 'QkvPositionedRopePagedKvWriteProductionAbiV2' \
  engine/device_step/fused_qkv_readonly_admit.mbt
rg -q 'ApprovedPagedAttentionReadOnlyRuntimeIdentity' engine/device_step/fused_qkv_readonly_types.mbt
rg -q 'QualifiedPagedKvWriteFallbackIdentity' engine/device_step/fused_qkv_readonly_types.mbt
rg -q 'adapt_paged_attention_readonly_runtime_identity' \
  release/luna_paged_attention_readonly_production/runtime_identity_adapter.mbt
rg -q 'adapt_qualified_paged_kv_write_fallback_identity' \
  release/luna_paged_kv_write_physical_evidence/runtime_identity_adapter.mbt
rg -q 'ContextDeviceIdentity' device/execution_types.mbt
rg -q 'Context::device_identity' device/execution_types.mbt
rg -q 'opened-context-device-identity' \
  release/luna_paged_attention_readonly_production/runtime_device.mbt
rg -q 'live_device : @device.ContextDeviceIdentity' \
  release/luna_paged_attention_readonly_production/runtime_device.mbt
rg -q 'rejects live-device substitution' \
  engine/device_step/fused_qkv_readonly_wbtest.mbt
rg -q 'pub fn select_fused_qkv_readonly_production_artifacts' \
  kernels/luna_paged_composite_artifact_admission/fused_production.mbt
rg -q 'pub fn admit_production_fused_qkv_readonly_attention_span' \
  engine/device_step/fused_qkv_readonly_production_admit.mbt
rg -q 'prepare_fused_qkv_readonly_resources' \
  engine/device_step/fused_qkv_readonly_prepare.mbt
rg -q 'replace_fused_qkv_readonly_span' \
  engine/device_step/fused_qkv_readonly_prepare.mbt
rg -q 'fused_qkv_readonly_span~' engine/device_worker/prepare.mbt
rg -q 'fused_qkv_readonly_span?' engine/device_step/paged_executor_prepare.mbt
rg -F -q 'let graph_policy = authenticated_bf16_graph_policy(execution)' \
  engine/device_worker/prepare.mbt
rg -q 'fused_qkv_readonly_artifacts?' engine/device_worker/prepare.mbt
rg -q 'adapt_fused_qkv_readonly_production_artifacts' \
  release/luna_fused_physical_evidence/production_manifest_admit.mbt
rg -q 'second-launch failure aborts ordered execution' \
  engine/device_step/fused_qkv_readonly_wbtest.mbt
rg -q 'rollback closes functions before modules' \
  engine/device_step/fused_qkv_readonly_wbtest.mbt

if rg -U -n \
  'FusedQkvReadOnlyAttentionCompositeRequirement::dispatch_authorized\([^)]*\)[^{]*\{[[:space:]]*false[[:space:]]*\}' \
  engine/device_step/fused_qkv_readonly_types.mbt >/dev/null; then
  echo "fused QKV dispatch remains a literal-false dead end" >&2
  exit 1
fi

if rg -n 'copy_(from|to)_fixed_host|dispatch_canary|@crypto|PagedOrderedEagerOnly' \
  engine/device_step/fused_qkv_readonly_prepare.mbt >/dev/null ||
  rg -n 'copy_(from|to)_fixed_host|PagedOrderedEagerOnly' \
  engine/device_step/fused_qkv_readonly_production_admit.mbt >/dev/null; then
  echo "production QKV execution gained diagnostics, crypto, or forced eager mode" >&2
  exit 1
fi

if rg -n 'LunaExternalSignatureApprovalReceipt|from_deployment_approval' \
  release/luna_paged_attention_readonly_production engine/device_step/fused_qkv_readonly_*.mbt
then
  echo "forgeable external approval receipt remains in readonly/composite boundary" >&2
  exit 1
fi

scripts/validate-production-release-import-boundary.sh

echo "fused QKV -> readonly attention composite boundary: pass"
