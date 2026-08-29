#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

scripts/validate-paged-graph-telemetry.sh
scripts/validate-luna-tile-parallel-specialization.sh
scripts/validate-luna-tensor-core-candidate.sh
scripts/validate-luna-tile-parallel-physical-campaign.sh
scripts/validate-luna-tensor-core-physical-campaign.sh
scripts/validate-luna-tensor-core-physical-evidence.sh
scripts/validate-luna-fused-artifact-admission.sh
scripts/validate-paged-attention-readonly-aot.sh

packages='kernels/luna_tile_ir kernels/luna_specialization kernels/luna_cuda_aot kernels/luna_capability_manifest kernels/luna_artifact_admission'

for package in $packages; do
  moon check "$package" --target native --deny-warn --warn-list +73
  moon test "$package" --target native --deny-warn --warn-list +73
  moon info "$package" --target native >/dev/null
done

if rg -n 'extern "c"|moonbitlang/async|vectie/lunaflux/(scheduler|service|internal/cuda)' $packages --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'Luna Phase5 offline/AOT boundary imports runtime authority' >&2
  exit 1
fi

if rg -n 'ed25519' moon.mod $packages --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'Luna Phase5 crossed the deployment-owned signature boundary' >&2
  exit 1
fi

if rg -n '(^|[^A-Za-z])(Jit|JIT|RuntimeCompile|CompilerHandle|CudaSource)([^A-Za-z]|$)' $packages --glob '*.mbt'; then
  printf '%s\n' 'Luna Phase5 public source contains a runtime compiler/JIT channel' >&2
  exit 1
fi

if rg -n 'LunaTileSemanticFamily|LunaEmbeddingV1|LunaRmsNormV1|LunaRotaryEmbeddingV1|LunaCausalAttentionV1|LunaGatedMlpV1' $packages --glob '*.mbt'; then
  printf '%s\n' 'Luna Phase5 exposes an unproved caller-selected semantic family' >&2
  exit 1
fi

if [ "$(rg -Fxc '  LunaValidatedEagerFallback' kernels/luna_capability_manifest/graph.mbt)" -ne 1 ]; then
  printf '%s\n' 'Luna graph disposition is no longer eager-only' >&2
  exit 1
fi

for required in \
  'pub fn LunaTileProgramInput::residual_add(' \
  'pub fn LunaTileProgram::residual_add_semantics(' \
  'pub struct LunaAuthenticatedExternalApproval {' \
  'pub fn admit_luna_external_approved_capability(' \
  'pub fn admit_luna_specialized_artifact(' \
  'pub fn specialize_luna_tile(' \
  'pub fn plan_luna_tile_cuda_aot_input(' \
  'pub fn lower_luna_residual_add_cuda_aot(' \
  'pub fn admit_luna_cuda_aot_output('; do
  if ! rg -F -q "$required" $packages --glob '*.mbt'; then
    printf 'Luna Phase5 required boundary missing: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'LunaExternalSignatureApprovalReceipt|from_deployment_approval' \
  kernels/luna_capability_manifest engine/execution_manifest_file \
  --glob '*.mbt'; then
  printf '%s\n' 'capability-free external approval construction remains' >&2
  exit 1
fi

production='engine/execution_manifest_file engine/device_step engine/device_worker engine/device_worker_bootstrap engine/worker_wire runtime/descriptor_file'
verifier_handoff_tests='tests/promotion_verifier_key_e2e kernels/luna_capability_manifest engine/worker_wire'

if rg -n 'vectie/lunaflux/kernels/luna_cuda_aot' $production \
  --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'production runtime imported the offline CUDA compiler boundary' >&2
  exit 1
fi

for package in $production; do
  moon check "$package" --target native --deny-warn --warn-list +73
  moon test "$package" --target native --deny-warn --warn-list +73
  moon info "$package" --target native >/dev/null
done

moon check tests/promotion_verifier_key_e2e \
  --target native --deny-warn --warn-list +73
moon test tests/promotion_verifier_key_e2e \
  --target native --deny-warn --warn-list +73

for required in \
  'pub fn admit_persisted_luna_specialization(' \
  'pub fn readmit_persisted_luna_specialization(' \
  'pub fn admit_persisted_luna_capability_manifest(' \
  '@luna_artifact_admission.admit_luna_specialized_artifact(' \
  'preflight_luna_approval_binding(claims.luna_authorization, luna_approval)' \
  'luna_authorization=execution.luna_authorization()' \
  'luna_approval=kernel.luna_approval()' \
  'DenseLlamaPagedAotV3'; do
  if ! rg -F -q "$required" $packages $production --glob '*.mbt'; then
    printf 'Luna Phase5 production composition missing: %s\n' "$required" >&2
    exit 1
  fi
done

for hostile in \
  'persisted Phase5 records reject truncation and trailing bytes' \
  'worker bootstrap digest retains exact untrusted Luna claim' \
  'optional promotion verifier distinguishes missing and invalid FD 7' \
  'verifier framing rejects malformed trailing and truncated input' \
  'startup verifier handoff fails closed on a substituted manifest' \
  'parent approval attestation rejects stale generation and substitution' \
  'v3 rejects mismatched approval before ordinary module open'; do
  if ! rg -F -q "$hostile" $packages $production $verifier_handoff_tests \
    --glob '*.mbt'; then
    printf 'Luna Phase5 hostile composition test missing: %s\n' "$hostile" >&2
    exit 1
  fi
done

for authenticated in \
  'fragmented exact verifier frame becomes one opaque handoff' \
  'startup verifier handoff is one-shot and wipes every key alias' \
  'parent approval attestation binds launch identity and consumes once'; do
  if ! rg -F -q "$authenticated" $verifier_handoff_tests --glob '*.mbt'; then
    printf 'Luna Phase5 authenticated startup composition test missing: %s\n' \
      "$authenticated" >&2
    exit 1
  fi
done

if rg -n 'pub fn LunaAotKernelAdmission::(new|from)' \
  kernels/luna_artifact_admission --glob '*.mbt' --glob 'pkg.generated.mbti'; then
  printf '%s\n' 'Luna execution authorization became publicly fabricable' >&2
  exit 1
fi

if rg -n 'Luna(Second|Parallel)(Executor|Catalog)|PhysicalLuna(Graph|Dispatch)' $production --glob '*.mbt'; then
  printf '%s\n' 'Luna Phase5 introduced a parallel executor/catalog or physical claim' >&2
  exit 1
fi

for file in $(rg --files $packages -g '*.mbt'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'Luna Phase5 source exceeds file budget: %s (%s)\n' "$file" "$lines" >&2
    exit 1
  fi
done


for file in \
  engine/execution_manifest_file/luna_load.mbt \
  engine/execution_manifest_file/schema_luna.mbt \
  engine/device_step/luna_authorization.mbt \
  engine/device_step/bootstrap_admit.mbt \
  engine/device_step/paged_executor_prepare.mbt \
  engine/worker_wire/bootstrap_source_codec.mbt \
  engine/worker_wire/bootstrap_source_types.mbt \
  engine/worker_wire/bootstrap_source_validate.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf 'Luna Phase5 integration source exceeds file budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done

scripts/test-luna-cuda-aot-builder.sh
scripts/validate-luna-positioned-paged-kv-write-aot.sh
scripts/validate-paged-kv-write-cuda-probe.sh
scripts/validate-paged-kv-write-physical-evidence.sh
scripts/validate-paged-kv-write-physical-campaign.sh
scripts/validate-luna-fused-grouped-runtime.sh

printf '%s\n' 'Luna Phase5 offline/AOT software gate passed'
