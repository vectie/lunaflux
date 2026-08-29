#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

fail_matches() {
  description=$1
  shift
  if matches=$(rg -n "$@" 2>/dev/null); then
    printf '%s\n%s\n' "$description" "$matches" >&2
    failed=1
  fi
}

# Scheduling policy is deliberately hardware-, transport-, and model-family
# agnostic. Add capabilities at the package boundary instead of exceptions.
if [ -d scheduler ]; then
  fail_matches \
    'scheduler has a forbidden dependency:' \
    --glob 'scheduler/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|device|internal/(cuda|nccl))(/|"|$)'
  fail_matches \
    'scheduler must not import serialized worker transport:' \
    --glob 'scheduler/**/moon.pkg' \
    '^\s*"vectie/lunaflux/engine/worker_wire"'
  fail_matches \
    'scheduler may import only canonical public model identity vocabulary:' \
    --pcre2 --glob 'scheduler/**/moon.pkg' \
    -U \
    'import\s*\{[^}]*"vectie/lunaflux/model/(?!spec"(?:,)?$)[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if [ -f scheduler/core/pkg.generated.mbti ] &&
    rg -n '@worker_wire|WorkerWire' scheduler/core/pkg.generated.mbti; then
    printf '%s\n' \
      'scheduler public interface leaks serialized worker transport' >&2
    failed=1
  fi
fi

if [ -d model ]; then
  fail_matches \
    'model has a forbidden dependency:' \
    --glob 'model/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|scheduler)(/|"|$)'
fi

# Correctness-reference packages are deliberately backend-independent. The
# offline interpreter may use model-family builders only from its test import.
if [ -d kernels/reference ]; then
  fail_matches \
    'reference kernels have a forbidden dependency:' \
    --glob 'kernels/reference/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|device|engine|internal|model|scheduler)(/|"|$)'
fi

if [ -d engine/reference ]; then
  fail_matches \
    'reference interpreter has a forbidden runtime dependency:' \
    --glob 'engine/reference/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|device|internal/(cuda|nccl)|scheduler)(/|"|$)'
  if ! rg -U -q \
    'import \{[^}]*"vectie/lunaflux/model/llama"[^}]*\} for "test"' \
    engine/reference/moon.pkg; then
    printf '%s\n' \
      'engine/reference model-family fixture dependency must remain test-only' >&2
    failed=1
  fi
fi

# AOT artifact files consume only the public pinned-root capability. They must
# not recreate path-based portable filesystem traversal or reach through the
# native implementation package.
if [ -d kernels/artifact_file ]; then
  fail_matches \
    'artifact_file must not import an internal filesystem ABI:' \
    --glob 'kernels/artifact_file/**/moon.pkg' \
    'vectie/lunaflux/internal/approved_fs'
  fail_matches \
    'artifact_file production imports must not use portable async filesystem IO:' \
    --pcre2 --glob 'kernels/artifact_file/**/moon.pkg' -U \
    'import\s*\{[^}]*"moonbitlang/async/fs"[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if rg -n \
    'pub (async )?fn (load|load_paged_kv)\(StringView' \
    kernels/artifact_file/pkg.generated.mbti; then
    printf '%s\n' \
      'artifact_file public loaders must consume an ApprovedRoot, not a string root' >&2
    failed=1
  fi
fi

# Model configuration-file admission is synchronous and capability-relative.
# It must not regress to ambient paths or portable async filesystem access.
if [ -d model/config_file ]; then
  fail_matches \
    'config_file must not import an internal filesystem ABI:' \
    --glob 'model/config_file/**/moon.pkg' \
    'vectie/lunaflux/internal/approved_fs'
  fail_matches \
    'config_file production imports must not use portable async filesystem IO:' \
    --pcre2 --glob 'model/config_file/**/moon.pkg' -U \
    'import\s*\{[^}]*moonbitlang/async(/fs)?[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if ! rg -q \
    '^pub fn load\(@approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRelativeLocator,' \
    model/config_file/pkg.generated.mbti; then
    printf '%s\n' \
      'config_file loader must consume ApprovedRoot and ApprovedRelativeLocator' >&2
    failed=1
  fi
  if rg -n '^pub async fn load|^pub fn load\([^)]*String(View)?' \
    model/config_file/pkg.generated.mbti; then
    printf '%s\n' \
      'config_file loader must remain synchronous and free of string paths' >&2
    failed=1
  fi
fi

# Offline reference-artifact admission is synchronous and descriptor-relative.
# Locator text is admitted only into one opaque source; the loader itself sees
# one caller-owned root and cannot recover ambient path authority.
if [ -d model/artifact ]; then
  fail_matches \
    'model/artifact must not import an internal filesystem ABI:' \
    --glob 'model/artifact/**/moon.pkg' \
    'vectie/lunaflux/internal/approved_fs'
  fail_matches \
    'model/artifact production imports must not use async filesystem IO:' \
    --pcre2 --glob 'model/artifact/**/moon.pkg' -U \
    'import\s*\{[^}]*moonbitlang/async(/fs)?[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if ! rg -q \
    '^pub fn load_reference_bundle\(@approved_fs\.ApprovedRoot, ArtifactSource, ExpectedDigests, ArtifactLimits\)' \
    model/artifact/pkg.generated.mbti; then
    printf '%s\n' \
      'reference artifact loader must consume ApprovedRoot and opaque ArtifactSource' >&2
    failed=1
  fi
  if rg -n \
    '^pub async fn load_reference_bundle|^pub fn load_reference_bundle\([^)]*String(View)?|ArtifactPaths' \
    model/artifact/pkg.generated.mbti; then
    printf '%s\n' \
      'reference artifact loader restored async or ambient path authority' >&2
    failed=1
  fi
  artifact_source_interface="$(sed -n \
    '/^pub struct ArtifactSource {$/,/^}$/p' \
    model/artifact/pkg.generated.mbti)"
  if [ -z "$artifact_source_interface" ] ||
    ! printf '%s\n' "$artifact_source_interface" | rg -q 'private fields' ||
    printf '%s\n' "$artifact_source_interface" | rg -q 'Debug|String|Locator|path'; then
    printf '%s\n' \
      'ArtifactSource must remain opaque and expose no locator text or Debug surface' >&2
    failed=1
  fi
fi

if [ -d cmd/lunaflux ]; then
  reference_source="$(sed -n \
    '/^fn run_reference(/,/^\/\/\/|/p' cmd/lunaflux/main.mbt)"
  if ! printf '%s\n' "$reference_source" | rg -U -q \
    'let artifact_limits = reference_artifact_limits\(\)[\s\S]*ApprovedRoot::open_absolute[\s\S]*load_reference_bundle[\s\S]*root\.close\(\) catch \{[\s\S]*raise error[\s\S]*root\.close\(\)[\s\S]*@llama_admission\.complete'; then
    printf '%s\n' \
      'reference CLI must prepare limits before root open and close root before execution/output' >&2
    failed=1
  fi
fi

if [ -d engine/device_worker_bootstrap ]; then
  fail_matches \
    'device_worker_bootstrap must not use ambient or async filesystem APIs:' \
    --pcre2 --glob 'engine/device_worker_bootstrap/**/moon.pkg' -U \
    'vectie/lunaflux/internal/approved_fs|import\s*\{[^}]*moonbitlang/async/fs[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if ! rg -q \
    '^pub fn prepare\(@worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, luna_parent_witness\? : @luna_capability_manifest\.LunaParentStartupApprovalWitness\?\) -> DeviceWorkerBootstrapPreparation raise DeviceWorkerBootstrapError$' \
    engine/device_worker_bootstrap/pkg.generated.mbti; then
    printf '%s\n' \
      'device_worker_bootstrap must consume only typed startup/source inputs' >&2
    failed=1
  fi
  if rg -n 'ApprovedRoot|ApprovedRelativeLocator|DeviceWeightFileInspection|PagedExecutionAdmission|DeviceWorkerPlan' \
    engine/device_worker_bootstrap/pkg.generated.mbti; then
    printf '%s\n' \
      'device_worker_bootstrap public interface leaks preparation evidence' >&2
    failed=1
  fi
fi

# The production scheduler blueprint accepts only resolver-owned capacity
# evidence. Keep the capacity record opaque so callers cannot forge a wider or
# internally inconsistent envelope with a record update.
if [ -f config/runtime_resolved/pkg.generated.mbti ]; then
  resolved_capacity_interface="$(sed -n \
    '/^pub struct ResolvedRuntimeCapacity {$/,/^}/p' \
    config/runtime_resolved/pkg.generated.mbti)"
  if ! printf '%s\n' "$resolved_capacity_interface" |
    rg -q '^  // private fields$'; then
    printf '%s\n' \
      'resolved runtime capacity must remain opaque construction evidence' >&2
    failed=1
  fi
  if printf '%s\n' "$resolved_capacity_interface" | rg -q '^  [a-zA-Z_][^/]* :'; then
    printf '%s\n' \
      'resolved runtime capacity must not expose forgeable record fields' >&2
    failed=1
  fi
fi

# Production worker supervision owns one root-bound process facade. Initial
# preparation accepts ordinary approved roots; replacement is zero-argument,
# and the service cannot receive the legacy fixture supervisor.
if [ -d engine/worker_process ] && [ -d engine/worker_service ]; then
  if ! rg -q \
    '^pub fn prepare_with_approved_executable\(@worker_executable_file\.WorkerExecutableAdmission, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot, parent_approval\? : @worker_wire\.WorkerParentApprovalSubjects\?\) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError$' \
    engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'root-bound worker preparation must acquire from ordinary ApprovedRoot owners' >&2
    failed=1
  fi
  if rg -n 'WorkerApprovedRoots' engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'worker-process public interface must not expose the retained root pair' >&2
    failed=1
  fi
  if ! rg -q '^  ModelRootBinding\(@approved_fs\.ApprovedFsError\)$' \
      engine/worker_process/pkg.generated.mbti ||
    ! rg -q '^  KernelRootBinding\(@approved_fs\.ApprovedFsError\)$' \
      engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'root-bound errors must retain a role-specific payload-safe binding cause' >&2
    failed=1
  fi
  if rg -n 'require_(absolute_identity|model_root_binding|kernel_root_binding)' \
      engine/worker_process/root_bound_recovery.mbt; then
    printf '%s\n' \
      'root-bound restart must continue from retained roots without label revalidation' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn new_fixture\(@core\.Scheduler, WorkerServiceBinding, @worker_process\.RootBoundWorkerProcessSupervisor\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'worker service fixture constructor must be explicitly named' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn prepare_owned_approved\(@core\.SchedulerBlueprint, WorkerServiceBinding, @worker_executable_file\.WorkerExecutableAdmission, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, @worker_process\.WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot, WorkerRestartBackoffPolicy\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'production worker service must construct scheduler and rooted process internally' >&2
    failed=1
  fi
  if rg -n '^pub fn [^(]*\([^)]*(@core\.Scheduler([,)]|$)|@worker_process\.RootBoundWorkerProcessSupervisor)' \
    engine/worker_service/pkg.generated.mbti |
    rg -v '^.*pub fn (new_fixture|prepare_owned_approved|prepare_owned_online)\('; then
    printf '%s\n' \
      'worker service must not expose another alias-taking owner constructor' >&2
    failed=1
  fi
  if rg -n '::(scheduler|process|roots)\(' engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'worker service must not expose scheduler, process, or root owners' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn WorkerService::restart\(Self\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' 'worker service restart must remain zero-argument' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn WorkerService::progress\(Self\) -> WorkerServiceProgress' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'worker service must own one nonblocking progress transition' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn WorkerService::drain_restart_forbidden\(Self\) -> @core\.InstanceLossDrain' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'restart-forbidden service must expose deterministic terminal drain' >&2
    failed=1
  fi
  if rg -n 'WorkerServiceDispatch|submit_next|complete_oldest|recover_oldest' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'worker service must not expose blocking fixture dispatch APIs' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn RootBoundWorkerProcessSupervisor::begin_exchange\(Self, @worker_protocol\.SubmittedSchedulePlan\) -> UInt64' \
    engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'root-bound worker process must own nonblocking plan exchange' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn RootBoundWorkerProcessSupervisor::progress_exchange\(Self\) -> RootBoundWorkerExchangeProgress' \
    engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'root-bound exchange must expose bounded explicit progress' >&2
    failed=1
  fi
  if rg -n '^pub struct RootBoundWorkerExchange|ExchangeCompleted\(' \
    engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'root-bound exchange state or completion capability must not be returned by progress' >&2
    failed=1
  fi
  if rg -n 'PendingFrame(Read|Write)|ChildProcess|FixedArray\[Byte\]' \
    engine/worker_process/pkg.generated.mbti; then
    printf '%s\n' \
      'worker-process public interface leaks private transport authority' >&2
    failed=1
  fi
fi

# Execution-manifest admission may consume only the public approved-root
# capability and must remain synchronous and path-typed.
if [ -d engine/execution_manifest_file ]; then
  fail_matches \
    'execution_manifest_file must not import an internal filesystem ABI:' \
    --glob 'engine/execution_manifest_file/**/moon.pkg' \
    'vectie/lunaflux/internal/approved_fs'
  fail_matches \
    'execution_manifest_file production imports must not use portable async filesystem IO:' \
    --pcre2 --glob 'engine/execution_manifest_file/**/moon.pkg' -U \
    'import\s*\{[^}]*"moonbitlang/async/fs"[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
  if ! rg -q \
    '^pub fn load_paged\(@approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRelativeLocator,' \
    engine/execution_manifest_file/pkg.generated.mbti; then
    printf '%s\n' \
      'execution_manifest_file loader must consume ApprovedRoot + ApprovedRelativeLocator' >&2
    failed=1
  fi
  if rg -n \
    '^pub async fn load_paged|^pub fn load_paged\([^)]*String(View)?' \
    engine/execution_manifest_file/pkg.generated.mbti; then
    printf '%s\n' \
      'execution_manifest_file loader must remain sync and free of string paths' >&2
    failed=1
  fi
fi

# Device-worker preparation consumes the previously admitted inspection and
# aggregate execution evidence. It must not restore the old decomposed handoff
# or perform a second source inspection after admission.
if [ -d engine/device_worker ]; then
  if ! rg -q \
    '^pub fn admit_plan\(startup~ : @worker_wire\.WorkerStartupContract, model~ : @spec\.LlamaModelMetadata, weight_inspection~ : @device_materialize\.DeviceWeightFileInspection, execution~ : @execution_manifest_file\.PagedExecutionAdmission, bootstrap_limits~ : @device_step\.DeviceBootstrapLimits(,|\))' \
    engine/device_worker/pkg.generated.mbti; then
    printf '%s\n' \
      'device_worker admission must consume aggregate execution and inspected-weight evidence' >&2
    failed=1
  fi
  if rg -n \
    '^pub fn admit_plan\([^)]*(blueprint|artifacts|weight_layout)~' \
    engine/device_worker/pkg.generated.mbti; then
    printf '%s\n' \
      'device_worker admission must not restore decomposed execution evidence' >&2
    failed=1
  fi
  if rg -n '@device_materialize\.inspect_file' \
    --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
    engine/device_worker; then
    printf '%s\n' \
      'device_worker preparation must not repeat weight-file inspection' >&2
    failed=1
  fi
fi

# The worker protocol has one authoritative fixed-capacity representation.
# Fixture-only heap records and row objects must not return to its public API.
if [ -f engine/worker_protocol/pkg.generated.mbti ]; then
  legacy_worker_protocol_types=$(rg -n \
    '^pub (struct (SchedulePlan|PrefillRow|DecodeRow|CompletionRecord|CompletionEntry) \{|\(all\) enum CompletionOutcome \{)' \
    engine/worker_protocol/pkg.generated.mbti 2>/dev/null || true)
  if [ -n "$legacy_worker_protocol_types" ]; then
    printf '%s\n%s\n' \
      'worker protocol exposes a removed heap-backed plan/completion type:' \
      "$legacy_worker_protocol_types" >&2
    failed=1
  fi
  for fixed_worker_protocol_type in \
    SchedulePlanBuffer SubmittedSchedulePlan SubmittedPrefillRow \
    SubmittedDecodeRow CompletionBuffer SubmittedCompletion \
    SubmittedCompletionEntry; do
    if ! rg -q \
      "^pub struct ${fixed_worker_protocol_type}( \\{|\\()" \
      engine/worker_protocol/pkg.generated.mbti; then
      printf '%s\n' \
        "worker protocol fixed surface is missing ${fixed_worker_protocol_type}" >&2
      failed=1
    fi
  done
  for fixed_worker_protocol_enum in CompletionEntryKind WorkerFailure; do
    if ! rg -q \
      "^pub\\(all\\) enum ${fixed_worker_protocol_enum} \\{" \
      engine/worker_protocol/pkg.generated.mbti; then
      printf '%s\n' \
        "worker protocol typed vocabulary is missing ${fixed_worker_protocol_enum}" >&2
      failed=1
    fi
  done
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
