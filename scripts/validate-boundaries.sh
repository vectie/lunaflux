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

# LunaFlux must remain independently buildable from every sibling product.
fail_matches \
  'forbidden MoonSuite source dependency:' \
  -i --glob 'moon.pkg' --glob 'moon.mod' \
  'lunanexa|moongate|moondesk|moonclaw|moontown'

# Contracts are vocabulary only and cannot depend on implementation packages.
# The inference contract reuses the canonical model identity owned by
# model/spec; duplicating that public value would make request/cache identity
# weaker than startup admission.
if [ -d contracts ]; then
  fail_matches \
    'contracts must not import implementation packages:' \
    --glob 'contracts/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|tokenizer|engine|scheduler|kv|prefix|kernels|device|internal)(/|"|$)'
  fail_matches \
    'contracts may import only the canonical public model identity vocabulary:' \
    --pcre2 --glob 'contracts/**/moon.pkg' \
    '^\s*"vectie/lunaflux/model/(?!spec"(?:,)?$)'
fi

# The public approved-filesystem facade is the sole production importer of the
# descriptor-owning native ABI. This keeps raw capability composition out of
# model, kernel, scheduler, and process packages until a dedicated fixed-FD
# spawn lease is designed.
approved_fs_imports=$(rg -l '"vectie/lunaflux/internal/approved_fs"' \
  --glob 'moon.pkg' --glob '!runtime/approved_fs/moon.pkg' 2>/dev/null || true)
if [ -n "$approved_fs_imports" ]; then
  printf '%s\n%s\n' \
    'internal approved filesystem ABI has unauthorized importers:' \
    "$approved_fs_imports" >&2
  failed=1
fi

approved_fs_capability_imports=$(rg -l \
  '"vectie/lunaflux/internal/approved_fs_capability"' --glob 'moon.pkg' \
  2>/dev/null | rg -v '^internal/(approved_fs|process)/moon.pkg$' || true)
if [ -n "$approved_fs_capability_imports" ]; then
  printf '%s\n%s\n' \
    'approved filesystem capability has unauthorized importers:' \
    "$approved_fs_capability_imports" >&2
  failed=1
fi

root_identity_calls=$(rg -l '\.require_absolute_identity\(' \
  --glob '*.mbt' 2>/dev/null | \
  rg -v '^(runtime/approved_fs|engine/worker_process)/' || true)
if [ -n "$root_identity_calls" ]; then
  printf '%s\n%s\n' \
    'approved root identity admission has unauthorized call sites:' \
    "$root_identity_calls" >&2
  failed=1
fi

worker_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  engine/worker_process --glob '*.mbt' --glob '!*_test.mbt' \
  --glob '!*_wbtest.mbt' 2>/dev/null || true)
worker_identity_count=$(printf '%s\n' "$worker_identity_calls" | \
  sed '/^$/d' | wc -l | tr -d ' ')
if [ "$worker_identity_count" -ne 2 ] ||
  printf '%s\n' "$worker_identity_calls" | \
    rg -v '^engine/worker_process/root_bound_binding\.mbt:'; then
  printf '%s\n%s\n' \
    'root identity must be checked exactly once per role during initial binding:' \
    "$worker_identity_calls" >&2
  failed=1
fi

for binding_role in model kernel; do
  helper="require_${binding_role}_root_binding"
  helper_files=$(rg -l "$helper" engine/worker_process --glob '*.mbt' \
    --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null || true)
  helper_file_count=$(printf '%s\n' "$helper_files" | sed '/^$/d' | \
    wc -l | tr -d ' ')
  helper_calls=$(rg -n "$helper\(" engine/worker_process --glob '*.mbt' \
    --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null | \
    rg -v '^engine/worker_process/root_bound_binding\.mbt:' || true)
  helper_call_count=$(printf '%s\n' "$helper_calls" | sed '/^$/d' | \
    wc -l | tr -d ' ')
  if [ "$helper_file_count" -ne 2 ] ||
    printf '%s\n' "$helper_files" | \
      rg -v '^engine/worker_process/root_bound(_binding)?\.mbt$' ||
    [ "$helper_call_count" -ne 1 ] ||
    ! printf '%s\n' "$helper_calls" | \
      rg -q '^engine/worker_process/root_bound\.mbt:'; then
    printf '%s\n%s\n' \
      "root identity $binding_role helper escaped initial preparation:" \
      "$helper_files" >&2
    failed=1
  fi
done

# The public monotonic-clock facade is the sole importer of its primitive-only
# native ABI. Higher layers consume the opaque runtime capability instead.
monotonic_clock_imports=$(rg -l \
  '"vectie/lunaflux/internal/monotonic_clock"' --glob 'moon.pkg' \
  --glob '!runtime/monotonic_clock/moon.pkg' 2>/dev/null || true)
if [ -n "$monotonic_clock_imports" ]; then
  printf '%s\n%s\n' \
    'internal monotonic clock ABI has unauthorized importers:' \
    "$monotonic_clock_imports" >&2
  failed=1
fi

if [ -d runtime/monotonic_clock ]; then
  fail_matches \
    'runtime monotonic clock must remain cycle-neutral:' \
    --glob 'runtime/monotonic_clock/moon.pkg' \
    'vectie/lunaflux/(contracts|device|engine|kernels|kv|model|prefix|scheduler|service|tokenizer)'
fi

# Scheduling policy is deliberately hardware-, transport-, and model-family
# agnostic. Add capabilities at the package boundary instead of exceptions.
if [ -d scheduler ]; then
  fail_matches \
    'scheduler has a forbidden dependency:' \
    --glob 'scheduler/**/moon.pkg' \
    '^\s*"vectie/lunaflux/(api|device|internal/cuda)(/|"|$)'
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
    '^\s*"vectie/lunaflux/(api|device|internal/cuda|scheduler)(/|"|$)'
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
    '^pub fn prepare\(@worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource\)' \
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
    '^pub fn prepare_with_approved_roots\(Bytes, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot\)' \
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
    '^pub fn prepare_owned\(@core\.SchedulerBlueprint, WorkerServiceBinding, Bytes, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, @worker_process\.WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'production worker service must construct scheduler and rooted process internally' >&2
    failed=1
  fi
  if rg -n '^pub fn [^(]*\([^)]*(@core\.Scheduler|@worker_process\.RootBoundWorkerProcessSupervisor)' \
    engine/worker_service/pkg.generated.mbti |
    rg -v '^.*pub fn (new_fixture|prepare_owned|prepare_owned_online)\('; then
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
    '^pub fn admit_plan\(startup~ : @worker_wire\.WorkerStartupContract, model~ : @spec\.LlamaModelMetadata, weight_inspection~ : @device_materialize\.DeviceWeightFileInspection, execution~ : @execution_manifest_file\.PagedExecutionAdmission, bootstrap_limits~ : @device_step\.DeviceBootstrapLimits\)' \
    engine/device_worker/pkg.generated.mbti; then
    printf '%s\n' \
      'device_worker admission must consume aggregate execution and inspected-weight evidence' >&2
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

if ! scripts/validate-service-boundaries.sh; then
  failed=1
fi
# Production foreign declarations have exactly four narrow owners: CUDA,
# approved descriptor-relative filesystem authority, and shell-free child
# process transport, plus monotonic time, each under its dedicated internal ABI
# package.
# Positive-controlled allocation harnesses and the exact approved-root child /
# parent E2E probes are the sole exceptions. Their narrow C shims inspect
# process state or generated allocation entry points and are not imported by a
# production package.
fail_matches \
  'production native declarations are only allowed under approved internal ABI packages:' \
  --glob '*.mbt' --glob '!internal/cuda/**' \
  --glob '!internal/approved_fs/**' \
  --glob '!internal/monotonic_clock/**' \
  --glob '!internal/process/**' \
  --glob '!tests/hot_path_alloc/**' \
  --glob '!tests/device_step_alloc/**' \
  --glob '!tests/device_worker_alloc/**' \
  --glob '!cmd/approved_root_echo/**' \
  --glob '!tests/approved_root_inheritance_e2e/**' \
  'extern\s+"[cC]"|#external'

# Internal ABI concrete types must never become part of a public package
# interface. Generated interfaces are authoritative for this boundary.
fail_matches \
  'public package interface leaks an internal ABI type:' \
  --glob 'pkg.generated.mbti' --glob '!internal/**' \
  'vectie/lunaflux/internal/'

root_owner_calls=$(rg -n '\.root_owner\(' --glob '*.mbt' || true)
if [ -n "$root_owner_calls" ] &&
  printf '%s\n' "$root_owner_calls" | rg -v '^engine/worker_process/'; then
  printf '%s\n' 'prepared approved-root owner sharing is restricted to worker_process' >&2
  failed=1
fi

fixture_constructor_calls=$(rg -n '@worker_service\.new_fixture\(' \
  --glob '*.mbt' || true)
if [ -n "$fixture_constructor_calls" ] &&
  printf '%s\n' "$fixture_constructor_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/)'; then
  printf '%s\n' 'WorkerService new_fixture escaped fixture/test scope' >&2
  failed=1
fi

online_lease_references=$(rg -n 'OnlineWorkerLease|take_online' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$online_lease_references" ] &&
  printf '%s\n' "$online_lease_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'online worker lease escaped its owned aggregate/test boundary' >&2
  failed=1
fi

online_transfer_calls=$(rg -n '\.take_online\(' --glob '*.mbt' || true)
if [ -n "$online_transfer_calls" ] &&
  printf '%s\n' "$online_transfer_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'owned online transfer escaped aggregate/test scope' >&2
  failed=1
fi

online_progress_status_calls=$(rg -n \
  '\.progress_status\(|\.progress_terminal_recovery_status\(' \
  --glob '*.mbt' || true)
if [ -n "$online_progress_status_calls" ] &&
  printf '%s\n' "$online_progress_status_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'sanitized online progress status escaped aggregate/test scope' >&2
  failed=1
fi

if ! rg -q '^pub fn OnlineWorkerLease::progress_status\(Self\) -> OnlineWorkerStep raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OnlineWorkerLease::progress_terminal_recovery_status\(Self, @worker_protocol.WorkerFailure\) -> OnlineTerminalRecoveryStatus$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'sanitized online progress status surface drifted' >&2
  failed=1
fi

scheduler_replacement_calls=$(rg -n \
  '\.replace_submitted_completion_with_failure\(' --glob '*.mbt' || true)
if [ -n "$scheduler_replacement_calls" ] &&
  printf '%s\n' "$scheduler_replacement_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^scheduler/core/|^engine/worker_service/)'; then
  printf '%s\n' 'scheduler invalid-completion replacement escaped recovery scope' >&2
  failed=1
fi

owned_online_prepare_calls=$(rg -n 'prepare_owned_online\(|\.take_prepared_online\(|\.commit_prepared_admission\(' \
  --glob '*.mbt' || true)
if [ -n "$owned_online_prepare_calls" ] &&
  printf '%s\n' "$owned_online_prepare_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'prepared online admission escaped aggregate/test scope' >&2
  failed=1
fi

exclusive_admission_references=$(rg -n \
  'PreparedExclusiveAdmission|prepare_exclusive_admission\(|commit_exclusive_admission\(|abort_exclusive_admission\(|has_exclusive_admission\(' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$exclusive_admission_references" ] &&
  printf '%s\n' "$exclusive_admission_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^scheduler/core/|^engine/worker_service/)'; then
  printf '%s\n' 'exclusive scheduler admission escaped worker-service/test scope' >&2
  failed=1
fi

prepared_clock_references=$(rg -n 'PreparedMonotonicRead' \
  --glob '*.mbt' --glob 'pkg.generated.mbti' || true)
if [ -n "$prepared_clock_references" ] &&
  printf '%s\n' "$prepared_clock_references" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^runtime/monotonic_clock/|^engine/worker_service/)'; then
  printf '%s\n' 'prepared monotonic reader escaped worker-service/test scope' >&2
  failed=1
fi

if ! rg -q '^pub struct PreparedExclusiveAdmission \{$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::prepare_exclusive_admission\(Self, TokenizedRequest\) -> PreparedExclusiveAdmission raise SchedulerError$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::commit_exclusive_admission\(Self, PreparedExclusiveAdmission, UInt64\) -> ExclusiveAdmissionCommit$' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -q '^pub fn Scheduler::abort_exclusive_admission\(Self, PreparedExclusiveAdmission\) -> Unit$' \
    scheduler/core/pkg.generated.mbti; then
  printf '%s\n' 'exclusive scheduler admission must remain opaque and exact' >&2
  failed=1
fi

raw_transfer_calls=$(rg -n '\.take_raw_ready\(' --glob '*.mbt' || true)
if [ -n "$raw_transfer_calls" ] &&
  printf '%s\n' "$raw_transfer_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/)'; then
  printf '%s\n' 'owned raw transfer escaped fixture/test scope' >&2
  failed=1
fi

if rg -n 'OwnedWorkerServicePreparation::take_ready|WorkerService::prepare_online_claim|OnlineWorkerLease::release' \
  engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'legacy owned/online extraction API remains public' >&2
  failed=1
fi

if [ -f service/framed_wire/pkg.generated.mbti ]; then
  if ! rg -q '^pub fn LunaFramedRequestStepBudget::new\(Int\) -> Self raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::new\(FramedWireLimits, LunaFramedRequestStepBudget\) -> Self$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::required_byte_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::required_int_cells\(FramedWireLimits\) -> UInt64 raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWorkspace::begin\(Self\) -> LunaFramedRequestWork raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::offer\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Int raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::progress\(Self\) -> LunaFramedRequestProgress raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestWork::take_view\(Self\) -> LunaFramedRequestView raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaFramedRequestView::release\(Self\) -> Unit raise FramedWireError$' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' 'Luna framed-request public contract drifted' >&2
    failed=1
  fi
  if [ "$(rg -c '^pub fn LunaFramedRequestStepBudget::' \
      service/framed_wire/pkg.generated.mbti)" != '2' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestWorkspace::' \
      service/framed_wire/pkg.generated.mbti)" != '4' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestWork::' \
      service/framed_wire/pkg.generated.mbti)" != '8' ] ||
    [ "$(rg -c '^pub fn LunaFramedRequestView::' \
      service/framed_wire/pkg.generated.mbti)" != '32' ]; then
    printf '%s\n' 'Luna framed-request method set drifted' >&2
    failed=1
  fi
  expected_luna_work_methods="$(printf '%s\n' \
    abort failure last_work_units offer progress state take_view \
    total_work_units | sort)"
  actual_luna_work_methods="$(sed -n \
    's/^pub fn LunaFramedRequestWork::\([^(:]*\).*/\1/p' \
    service/framed_wire/pkg.generated.mbti | sort)"
  if [ "$actual_luna_work_methods" != "$expected_luna_work_methods" ]; then
    printf '%s\n' 'Luna framed-request work authority surface drifted' >&2
    failed=1
  fi
  expected_luna_view_methods="$(printf '%s\n' \
    cache_permission cache_scope_byte_at cache_scope_length \
    content_digest_byte_at context_ceiling deadline_millis has_top_k \
    has_top_p inference_limits input_byte_at input_kind input_length \
    input_token_at length max_new_tokens plan_digest_byte_at \
    protocol_version_wire release request_id_value sampling_mode \
    sampling_seed_value sampling_temperature stop_string_byte_at \
    stop_string_count stop_string_length stop_token_at stop_token_count \
    stream_preference top_k top_p trace_byte_at trace_length | sort)"
  actual_luna_view_methods="$(sed -n \
    's/^pub fn LunaFramedRequestView::\([^(:]*\).*/\1/p' \
    service/framed_wire/pkg.generated.mbti | sort)"
  luna_request_private_count="$(rg -c --pcre2 -U \
    'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View) \{\n  // private fields\n\}' \
    service/framed_wire/pkg.generated.mbti)"
  if [ "$actual_luna_view_methods" != "$expected_luna_view_methods" ] ||
    [ "$luna_request_private_count" != '4' ] ||
    rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::.*-> .*(FixedArray|Array\[|ArrayView|ReadOnlyArray|Bytes|String|GenerateRequest|ValidatedRequestFrame)' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n '^pub fn LunaFramedRequest(StepBudget|Workspace|Work|View)::(owner|epoch|storage|workspace)\(' \
      service/framed_wire/pkg.generated.mbti ||
    rg -n 'pub struct LunaFramedRequest(StepBudget|Workspace|Work|View).*derive\([^)]*Debug' \
      service/framed_wire/pkg.generated.mbti; then
    printf '%s\n' \
      'Luna framed-request capability opacity or scalar view drifted' >&2
    failed=1
  fi
fi

if ! rg -q '^pub fn OwnedWorkerServicePreparation::take_raw_ready\(Self\) -> WorkerService raise OwnedWorkerServicePreparationError$' \
  engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_online\(Self, @monotonic_clock.MonotonicClock\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_prepared_online\(Self\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'owned preparation must expose exact raw and online transfers' >&2
  failed=1
fi

if ! rg -q '^pub fn OnlineWorkerLease::retire_terminal_request\(Self\) -> Unit raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OnlineWorkerLease::shutdown_clean_empty\(Self\) -> Unit raise WorkerServiceError$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' \
    'online worker lease must expose exact persistent retire and empty-shutdown seams' >&2
  failed=1
fi

if rg -n 'OnlineWorkerLease::(scheduler|process|handle|request_id|request_generation|publication)' \
  engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'online worker lease exposes raw owner or identity evidence' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  rg -n '(^|[^A-Za-z])WorkerService([^A-Za-z]|$)|OnlineWorkerLease|AdmittedRequest|ReceivedRequest|TokenizerSpec|LunaPreparedRequestClaim|TokenizedRequest|RequestHandle|SchedulerPublication|IncrementalOutput' service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'online session aggregate must not return its lower owners' >&2
  failed=1
fi

if rg -n 'framed_wire|FramedWireLimits|CanonicalEventWriter|EventFrameBuffer|ValidatedEventFrame|LunaOnlineWireFailure' \
  service/online_session --glob '*.mbt' --glob 'moon.pkg' --glob '*.mbti'; then
  printf '%s\n' \
    'online session must own semantic Luna events, not framed-wire state' >&2
  failed=1
fi

if ! rg -U -q \
  'fn begin_foundation_session[\s\S]*LunaFramedEventAdapter::new[\s\S]*let admission = instance\.begin\(prepared\)' \
  tests/worker_service_e2e/online_session_harness.mbt; then
  printf '%s\n' \
    'online driver must preflight its fallible event adapter before begin' >&2
  failed=1
fi

if rg -n '@request_admission\.(admit|prepare_luna_request)|@tokenizer\.TokenizerSpec|priv tokenizer[[:space:]]*:' \
  service/online_session --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' \
    'online instance must consume prepared requests without tokenizer work/state' >&2
  failed=1
fi

if rg -q '^  OutputFailure$' service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'online output failure remains a typed aggregate error' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  { ! rg -q '^pub fn prepare_owned_luna_online_instance\(' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn prepare_owned_luna_online_instance\(@tokenizer\.TokenizerDigest, @spec\.ModelIdentity, @inference\.InferenceLimits, @core\.SchedulerBlueprint, @worker_service\.WorkerServiceBinding, Bytes, @worker_wire\.WorkerStartupContract, @worker_wire\.EncodedBootstrapSource, @worker_process\.WorkerProcessLimits, @approved_fs\.ApprovedRoot, @approved_fs\.ApprovedRoot\) -> LunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError$' service/online_session/pkg.generated.mbti ||
    ! rg -q --pcre2 -U '^pub struct LunaOnlineRequestTicket \{\n  // private fields\n\} derive\(Eq\)$' service/online_session/pkg.generated.mbti ||
    ! rg -q --pcre2 -U '^pub struct LunaOnlineInstanceAdmission \{\n  // private fields\n\}$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstanceAdmission::kind\(Self\) -> LunaOnlineInstanceAdmissionKind$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstanceAdmission::ticket\(Self\) -> LunaOnlineRequestTicket raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::begin\(Self, @request_admission\.LunaPreparedRequest\) -> LunaOnlineInstanceAdmission raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_terminalization\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_request_retirement\(Self, LunaOnlineRequestTicket\) -> LunaOnlineInstanceRetirementProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::take_event\(Self, LunaOnlineRequestTicket\) -> LunaOnlineEventCredit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineEventCredit::view\(Self\) -> @luna_event\.LunaEventView raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineEventCredit::ack\(Self\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::begin_drain\(Self\) -> Unit$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::progress_shutdown\(Self\) -> LunaOnlineInstanceShutdownProgress raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::request_cancel\(Self, LunaOnlineRequestTicket\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn LunaOnlineInstance::check_deadline\(Self, LunaOnlineRequestTicket\) -> Unit raise LunaOnlineInstanceError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^  LunaOnlineInstanceRequestRetirementRequired$' service/online_session/pkg.generated.mbti; }; then
  printf '%s\n' 'persistent Luna online instance surface drifted' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  { ! rg -q --pcre2 -U \
      'pub struct LunaOnlineEventCredit \{\n  // private fields\n\}' \
      service/online_session/pkg.generated.mbti ||
    rg -n '^pub fn LunaOnlineInstance::(has_event|event_length|copy_event_to|ack_event)\(' \
      service/online_session/pkg.generated.mbti ||
    rg -n '^pub fn LunaOnlineEventCredit::(instance|ticket|event_epoch|epoch|raw)\(' \
      service/online_session/pkg.generated.mbti; }; then
  printf '%s\n' \
    'Luna online event credit must remain opaque with only view/ACK authority' >&2
  failed=1
fi

if rg -n 'AdmittedRequest|^pub fn admit\(' \
  service/request_admission/pkg.generated.mbti service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'legacy admitted-request/tokenizer admission API remains public' >&2
  failed=1
fi

if ! rg -q '^pub fn LunaPreparedRequest::take_claim\(Self\) -> LunaPreparedRequestClaim raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  rg -n 'claim_scheduler_request|LunaPreparedRequest::scheduler_request' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna prepared shell must expose only destructive claim transfer' >&2
  failed=1
fi

if rg -n --pcre2 -U \
    'pub struct (LunaPreparedRequest|LunaPreparedRequestClaim) \{\n  (?!// private fields)|pub struct LunaPreparedRequest(?:Claim)? \{(?s:[^}]*)\} derive\([^)]*Debug' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna prepared authority must remain opaque and non-Debug' >&2
  failed=1
fi

if [ "$(rg -c '^pub fn LunaPreparedRequest::' \
    service/request_admission/pkg.generated.mbti)" != '8' ] ||
  ! rg -q '^pub fn LunaPreparedRequest::discard\(Self\) -> Unit raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  rg -n --pcre2 \
    '^pub fn LunaPreparedRequest::(receipt|receipt_at_millis|timestamp|deadline|admission_deadline|scheduler_request|is_stop_token|push_token|push_token_into|push_token_into_status|finish_into|finish_into_status|output_finished|output_stopped)\(' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna prepared shell surface escaped binding preflight/destructive transfer' >&2
  failed=1
fi

if [ "$(rg -c '^pub fn LunaPreparedRequestClaim::' \
    service/request_admission/pkg.generated.mbti)" != '5' ] ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::scheduler_request\(Self\) -> @core\.TokenizedRequest raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::release\(Self\) -> Unit raise RequestAdmissionError$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::is_stop_token\(Self, Int\) -> Bool$' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::push_token_into_status\(' \
    service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaPreparedRequestClaim::finish_into_status\(' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna prepared claim surface escaped scheduler/output/release ownership' >&2
  failed=1
fi

if rg -n --pcre2 \
    '^pub fn .*@inference\.(?:LunaRequestSemantic(?:Lease|View|Storage|Work|Write)|LunaRequestStopToken(?:View|RetentionSlot)|StopConditions|CachePolicy|CacheScope)|^pub fn LunaPreparedRequestClaim::.*(?:Array\[String\]|ReadOnlyArray\[String\]|StringView|\bepoch\b)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'request admission leaked semantic retention or claim stop/cache representation authority' >&2
  failed=1
fi

claim_scheduler_consumers="$(rg -l \
  'claim\.scheduler_request\(\)' \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' \
  --glob '!tests/**' . | sed 's#^\./##' | sort || true)"
if [ "$claim_scheduler_consumers" != \
  'service/online_session/admission.mbt' ]; then
  printf '%s\n' \
    'prepared scheduler-request borrow escaped the online admission bridge' >&2
  failed=1
fi

direct_claim_releases="$(rg -n 'claim\.release\(\)' \
  service/online_session --glob '*.mbt' | wc -l | tr -d ' ')"
lifecycle_claim_releases="$(rg -n 'self\.release_request_claim\(\)' \
  service/online_session/lifecycle.mbt | wc -l | tr -d ' ')"
if [ "$direct_claim_releases" != '2' ] ||
  [ "$lifecycle_claim_releases" != '2' ] ||
  ! rg -q --pcre2 -U \
    'self\.lease_owner\(\)\.admit\(claim\.scheduler_request\(\)\) catch \{[\s\S]*try! claim\.release\(\)[\s\S]*try! self\.events\.discard\(\)' \
    service/online_session/admission.mbt ||
  ! rg -q --pcre2 -U \
    'fn LunaOnlineInstance::close_terminal_owner[\s\S]*lease\.close_terminal\(\)[\s\S]*lease\.retry_close_terminal\(\)[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.reset_request\(\)' \
    service/online_session/lifecycle.mbt ||
  ! rg -q --pcre2 -U \
    'lease\.retire_terminal_request\(\) catch[\s\S]*self\.release_request_claim\(\)[\s\S]*self\.reset_request\(\)' \
    service/online_session/lifecycle.mbt; then
  printf '%s\n' \
    'online claim release must follow lower rejection, terminal close, or healthy retirement exactly once' >&2
  failed=1
fi

if rg -n --pcre2 -U \
    'pub struct (LunaRequestPreparationPool|LunaRequestPreparationAdmission|LunaRequestPreparationWork|LunaRequestPreparationStepBudget|LunaRequestPreparationWorkLimit|LunaRequestPreparationStorageBudget) \{\n  (?!// private fields)' \
    service/request_admission/pkg.generated.mbti ||
  rg -n \
    '^pub fn (LunaPreparedRequest|LunaPreparedRequestClaim|LunaRequestPreparation[^:]*)::.*(LunaTokenBuffer|TokenBuffer|LunaIncrementalOutput(Workspace|Work|Lease)|Array\[Int\]|ArrayView\[Int\]|ReadOnlyArray\[Int\])' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'Luna request-preparation capabilities leaked raw storage or representation' >&2
  failed=1
fi

if ! rg -q '^pub fn LunaRequestPreparationPool::try_submit\(Self, ReceivedRequest\) -> LunaRequestPreparationAdmission raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationPool::try_begin_luna_framed\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> LunaRequestPreparationAdmission raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationPool::progress\(Self\) -> LunaRequestPreparationPoolProgress$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationAdmission::consumed_bytes\(Self\) -> Int$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::offer_luna_framed\(Self, FixedArray\[Byte\], source_offset~ : Int, length~ : Int\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::take_prepared\(Self\) -> LunaPreparedRequest raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::last_work_units\(Self\) -> Int raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti ||
  ! rg -q '^pub fn LunaRequestPreparationWork::total_work_units\(Self\) -> UInt64 raise RequestAdmissionError$' service/request_admission/pkg.generated.mbti; then
  printf '%s\n' 'Luna preparation submit/progress/take surface drifted' >&2
  failed=1
fi

if rg -n \
    '^pub fn .*(@framed_wire\.LunaFramedRequest(Workspace|Work|View)|RequestReceipt|@inference\.AdmissionDeadline)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'direct framed preparation leaked scanner, receipt, or deadline authority' >&2
  failed=1
fi

if rg -n \
    '^pub (struct|enum) (Luna.*Framed.*(Receipt|Admission)|RequestReceipt)' \
    service/request_admission/pkg.generated.mbti; then
  printf '%s\n' \
    'direct framed preparation introduced a parallel receipt/admission authority' >&2
  failed=1
fi

if rg -n \
    'GenerateRequest::new|Input::(from_utf8|from_token_ids)|TextInput::|StopConditions::new|CachePolicy::new|materialize_luna|@utf8\.encode|Bytes::make|Map::|HashMap' \
    service/request_admission/luna_framed_receipt.mbt \
    service/request_admission/pool_framed_progress.mbt \
    service/request_admission/pool_output_progress.mbt; then
  printf '%s\n' \
    'direct framed preparation reintroduced object materialization' >&2
  failed=1
fi

if rg -n 'moonbitlang/async|async fn|socket|listener|worker_(process|service)|device_|approved_fs' \
    service/request_admission/pool*.mbt; then
  printf '%s\n' \
    'Luna preparation pool acquired async/socket/process/device authority' >&2
  failed=1
fi

for required_authority_interface in \
  contracts/inference/pkg.generated.mbti \
  model/spec/pkg.generated.mbti \
  scheduler/core/pkg.generated.mbti; do
  if [ ! -f "$required_authority_interface" ]; then
    printf '%s\n' \
      "required authority interface ${required_authority_interface} is missing" >&2
    failed=1
  fi
done

assert_self_factory_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg \
    "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
    "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public Self factory allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_escape_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$({
    rg "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
      "$interface" || true
    rg --pcre2 \
      "^pub fn .* -> .*(?<![A-Za-z0-9_])${authority_type}(?![A-Za-z0-9_]).*$" \
      "$interface" || true
  } | sed 's/^pub fn //; s/(.*$//' | sort -u)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public authority escape allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_method_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg "^pub fn ${authority_type}::" "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public method allowlist drifted" >&2
    failed=1
  fi
}

for request_authority in \
    AdmissionDeadline \
    CachePolicy \
    CacheScope \
    DeadlineBudget \
    EffectiveLimits \
    GenerateRequest \
    InferenceLimits \
    ProtocolVersion \
    RequestId \
    SamplingParameters \
    SamplingSeed \
    StopConditions \
    TextInput \
    TokenBuffer \
    TraceCorrelation; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${request_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference request authority ${request_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline from_receipt
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope from_ascii
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget from_milliseconds
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti Input 'from_token_ids
from_utf8'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'from_wire
v1'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti RequestId new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'greedy
sample'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'new
token_only'
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TextInput from_utf8
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer new
assert_self_factory_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation from_ascii
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti AdmissionDeadline AdmissionDeadline::from_receipt
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CachePolicy 'CachePolicy::new
GenerateRequest::cache'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti CacheScope 'CachePolicy::scope
CacheScope::from_ascii'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti DeadlineBudget 'DeadlineBudget::from_milliseconds
GenerateRequest::deadline'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti EffectiveLimits 'AcceptedEvent::effective_limits
EffectiveLimits::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti GenerateRequest GenerateRequest::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti InferenceLimits 'InferenceLimits::new
LunaRequestSemanticView::inference_limits'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti Input 'GenerateRequest::input
Input::from_token_ids
Input::from_utf8'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti ProtocolVersion 'GenerateRequest::protocol_version
ProtocolVersion::from_wire
ProtocolVersion::v1'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti RequestId 'GenerateRequest::request_id
RequestId::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingParameters 'GenerateRequest::sampling
SamplingParameters::greedy
SamplingParameters::sample'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti SamplingSeed 'GenerateRequest::sampling_seed
SamplingSeed::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti StopConditions 'GenerateRequest::stops
StopConditions::new
StopConditions::token_only'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TextInput TextInput::from_utf8
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TokenBuffer 'LunaTokenBufferLease::token_buffer
TokenBuffer::new'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti TraceCorrelation 'GenerateRequest::trace
TraceCorrelation::from_ascii'
if ! rg -q --pcre2 -U \
    '^pub\(all\) enum Input \{\n  TokenIds\(TokenBuffer\)\n  Text\(TextInput\)\n\}$' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference Input authority variants drifted' >&2
  failed=1
fi
if ! rg -F -x -q \
    'pub fn Input::from_token_ids(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn Input::from_utf8(BytesView, InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn GenerateRequest::input(Self) -> Input' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn TokenBuffer::new(ArrayView[Int], InferenceLimits) -> Self raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn LunaTokenBufferLease::token_buffer(Self) -> TokenBuffer raise InferenceContractError' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference input and token authority signatures drifted' >&2
  failed=1
fi
if rg -q --pcre2 -U \
    '^pub struct (CachePolicy|CacheScope|GenerateRequest|SamplingSeed|StopConditions|TextInput|TokenBuffer|TraceCorrelation) \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'payload-bearing inference request authority became debuggable' >&2
  failed=1
fi

for semantic_authority in \
    LunaRequestSemanticFailure \
    LunaRequestSemanticLease \
    LunaRequestSemanticProgress \
    LunaRequestSemanticState \
    LunaRequestSemanticStepBudget \
    LunaRequestSemanticStopTokenStatus \
    LunaRequestSemanticStorage \
    LunaRequestSemanticView \
    LunaRequestSemanticWork \
    LunaRequestSemanticWrite; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${semantic_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference semantic authority ${semantic_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequestSemantic[^ ]* \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority became debuggable' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn LunaRequestSemantic[^:]*::.* -> .*\b(Array|ArrayView|ReadOnlyArray|FixedArray|Bytes|BytesView|String|StringView)\b' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority exposed a raw payload collection' >&2
  failed=1
fi
if rg -n '^pub fn LunaRequestSemantic[^:]*::.*epoch' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference semantic authority exposed a raw epoch' >&2
  failed=1
fi
if rg 'LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)' \
    contracts/inference/pkg.generated.mbti | \
    rg -n -v '^pub (struct|fn) '; then
  printf '%s\n' \
    'inference semantic authority was embedded in another public type' >&2
  failed=1
fi
if rg -n --pcre2 \
    '^pub fn (?!LunaRequestSemantic[^:]*::)[^(]*\([^)]*LunaRequestSemantic(Failure|Lease|Progress|State|StepBudget|StopTokenStatus|Storage|View|Work|Write)[^)]*\)' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' \
    'inference semantic authority gained a non-owner public consumer' >&2
  failed=1
fi

assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure 'field
index
issue'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease 'release
stop_token_view
view'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress 'is_failed
is_pending
is_ready'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState 'is_failed
is_ready
is_validating'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget 'new
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus 'is_absent
is_present
is_stale
work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage 'begin
new
required_byte_cells
required_int_cells'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView 'cache_permission
cache_scope_byte_at
cache_scope_length
inference_limits
is_stop_token
stop_string_byte_at
stop_string_count
stop_string_length
stop_token_at
stop_token_count'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork 'abort
failure
last_work_units
progress
state
take_lease
total_work_units'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite 'abort
finish
push_cache_scope_byte
push_stop_string_byte
push_stop_string_length
push_stop_token'

assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticFailure \
  LunaRequestSemanticWork::failure
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticLease \
  LunaRequestSemanticWork::take_lease
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticProgress \
  LunaRequestSemanticWork::progress
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticState \
  LunaRequestSemanticWork::state
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStepBudget \
  LunaRequestSemanticStepBudget::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStopTokenStatus \
  $'LunaRequestSemanticView::is_stop_token\nLunaRequestStopTokenRetentionSlot::is_stop_token\nLunaRequestStopTokenView::is_stop_token'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticStorage \
  LunaRequestSemanticStorage::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticView \
  LunaRequestSemanticLease::view
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWork \
  LunaRequestSemanticWrite::finish
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestSemanticWrite \
  LunaRequestSemanticStorage::begin

for stop_authority in \
    LunaRequestStopTokenRetentionSlot \
    LunaRequestStopTokenView; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${stop_authority} \\{\\n  // private fields\\n\\}" \
      contracts/inference/pkg.generated.mbti; then
    printf '%s\n' \
      "inference token-stop authority ${stop_authority} is not opaque" >&2
    failed=1
  fi
done
if rg -q --pcre2 -U \
    '^pub struct LunaRequestStopToken[^ ]* \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti; then
  printf '%s\n' 'inference token-stop authority became debuggable' >&2
  failed=1
fi
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot 'is_live
is_stop_token
new
release'
assert_authority_method_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView 'is_live
is_stop_token
retain_into'
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenRetentionSlot \
  LunaRequestStopTokenRetentionSlot::new
assert_authority_escape_allowlist \
  contracts/inference/pkg.generated.mbti LunaRequestStopTokenView \
  LunaRequestSemanticLease::stop_token_view

for model_identity_authority in \
  ContentDigest \
  LlamaModelMetadata \
  LlamaModelSpec \
  ModelIdentity \
  PlanDigest; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${model_identity_authority} \\{\\n  // private fields\\n\\}" \
      model/spec/pkg.generated.mbti; then
    printf '%s\n' \
      "model identity authority ${model_identity_authority} is not opaque" >&2
    failed=1
  fi
done
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ContentDigest from_sha256
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata from_verified_content
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity new
assert_self_factory_allowlist \
  model/spec/pkg.generated.mbti PlanDigest from_sha256
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ContentDigest 'ContentDigest::from_sha256
ModelIdentity::content'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelMetadata LlamaModelMetadata::from_verified_content
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti LlamaModelSpec 'LlamaModelMetadata::spec
LlamaModelSpec::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti ModelIdentity 'LlamaModelMetadata::identity
ModelIdentity::new'
assert_authority_escape_allowlist \
  model/spec/pkg.generated.mbti PlanDigest 'LlamaModelSpec::canonical_paged_plan_digest
LlamaModelSpec::canonical_plan_digest
ModelIdentity::plan
PlanDigest::from_sha256'
if rg -n '^pub(\(all\))? type' \
    contracts/inference/pkg.generated.mbti model/spec/pkg.generated.mbti; then
  printf '%s\n' \
    'authority packages must not hide authority returns behind public aliases' >&2
  failed=1
fi
if ! rg -q '^pub fn ContentDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn PlanDigest::from_sha256\(String\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn ModelIdentity::new\(ContentDigest, PlanDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelMetadata::from_verified_content\(LlamaModelSpec, ContentDigest\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti ||
  ! rg -q '^pub fn LlamaModelSpec::new\(vocabulary_size~ : Int, hidden_size~ : Int, intermediate_size~ : Int, layer_count~ : Int, attention_head_count~ : Int, kv_head_count~ : Int, context_length~ : Int, rope_theta~ : Double, rms_norm_epsilon\? : Double, weight_dtype~ : ModelDType, kv_dtype~ : ModelDType\) -> Self raise ModelSpecError$' model/spec/pkg.generated.mbti; then
  printf '%s\n' 'model authority factory exclusivity drifted' >&2
  failed=1
fi

if rg -n '^pub fn TokenBuffer::token_ids\(|^pub fn TokenizedRequest::(input_tokens|input_token_at)\(' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
  rg -n --pcre2 -U \
    'pub struct (TokenBuffer|TokenizedRequest) \{\n  (?!// private fields)' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti ||
  rg -n --pcre2 -U \
    'pub struct (TokenBuffer|TokenizedRequest) \{(?s:[^}]*)\} derive\([^)]*Debug' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
  printf '%s\n' \
    'scheduler token request leaked raw token arrays or debuggable representation' >&2
  failed=1
fi

token_bound_surface="$(rg \
  '^pub fn (LunaTokenBufferBoundStatus::|TokenBuffer::maximum_token_status\()' \
  contracts/inference/pkg.generated.mbti || true)"
if ! rg -q --pcre2 -U \
    '^pub struct LunaTokenBufferBoundStatus \{\n  // private fields\n\}$' \
    contracts/inference/pkg.generated.mbti ||
  rg -q --pcre2 -U \
    '^pub struct LunaTokenBufferBoundStatus \{(?s:[^}]*)\} derive\([^)]*(?:@debug\.)?Debug' \
    contracts/inference/pkg.generated.mbti ||
  [ "$token_bound_surface" != \
  $'pub fn LunaTokenBufferBoundStatus::is_out_of_range(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_stale(Self) -> Bool\npub fn LunaTokenBufferBoundStatus::is_within_bound(Self) -> Bool\npub fn TokenBuffer::maximum_token_status(Self, Int) -> LunaTokenBufferBoundStatus' ]; then
  printf '%s\n' \
    'authenticated Luna token-maximum bound opacity or surface drifted' >&2
  failed=1
fi

if rg -n \
    '^pub fn TokenBuffer::.*maximum.* -> Int$|^pub fn TokenizedRequest::input_maximum_token_status\(' \
    contracts/inference/pkg.generated.mbti scheduler/core/pkg.generated.mbti; then
  printf '%s\n' \
    'raw token maximum or scheduler forwarding authority became public' >&2
  failed=1
fi
if ! rg -F -x -q \
    'pub fn TokenizedRequest::new(request_id~ : @inference.RequestId, model_identity~ : @spec.ModelIdentity, input_tokens~ : @inference.TokenBuffer, sampling~ : @inference.SamplingParameters, sampling_seed~ : @inference.SamplingSeed, stop_tokens~ : @inference.LunaRequestStopTokenView, stream_preference~ : @inference.StreamPreference, effective_limits~ : @inference.EffectiveLimits, admission_deadline~ : @inference.AdmissionDeadline) -> Self raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn PreparedAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  ! rg -F -x -q \
    'pub fn PreparedExclusiveAdmission::retain_stop_tokens_into(Self, @inference.LunaRequestStopTokenRetentionSlot) -> Unit raise SchedulerError' \
    scheduler/core/pkg.generated.mbti ||
  rg -n '^pub fn TokenizedRequest::(stops|cache|stop_tokens|semantic)' \
    scheduler/core/pkg.generated.mbti ||
  rg -n 'LunaRequestSemanticView|StopConditions|CachePolicy' \
    scheduler/core/pkg.generated.mbti; then
  printf '%s\n' 'scheduler token-only semantic authority surface drifted' >&2
  failed=1
fi

if rg -n \
    '@inference\.(LunaRequestSemanticView|StopConditions|CachePolicy)|\.stops\(\)|\.cache\(\)|stop_string_(count|length|byte_at)|cache_scope_(length|byte_at)|token_only\(' \
    scheduler/core --glob '*.mbt' --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt' ||
  ! rg -q \
    'priv stop_tokens : @inference\.LunaRequestStopTokenView' \
    scheduler/core/request.mbt ||
  ! rg -q \
    '@inference\.LunaRequestStopTokenRetentionSlot' \
    scheduler/core/types.mbt ||
  ! rg -F -q \
    'retention.is_stop_token(token)' \
    scheduler/core/completion_preflight.mbt; then
  printf '%s\n' \
    'scheduler source escaped the narrow Luna stop-token projection boundary' >&2
  failed=1
fi

if rg -n \
  'prepare_owned_session|OnlineSession(::|Preparation|Progress|Cleanup|Error|Failure|Rule|\s*\{)' \
  service/online_session tests/worker_service_e2e/online_session_*.mbt; then
  printf '%s\n' 'removed one-shot OnlineSession API returned' >&2
  failed=1
fi

if rg -n \
  '^pub\(all\) struct LunaOnlineRequestTicket|^pub struct LunaOnlineRequestTicket\(|^pub fn LunaOnlineRequestTicket::(value|epoch|id|raw)\(|^pub fn LunaOnlineInstanceAdmission::(epoch|raw|status)\(' \
  service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'Luna online request tickets must remain opaque scalar authority' >&2
  failed=1
fi

persistent_lower_calls=$(rg -n \
  '\.(retire_terminal_request|shutdown_clean_empty|retry_close_empty)\(' \
  --glob '*.mbt' || true)
if [ -n "$persistent_lower_calls" ] &&
  printf '%s\n' "$persistent_lower_calls" |
    rg -v '(^tests/|_test\.mbt:|_wbtest\.mbt:|^engine/worker_service/|^service/online_session/)'; then
  printf '%s\n' 'persistent worker retire/shutdown authority escaped aggregate scope' >&2
  failed=1
fi

# Production code is MoonBit plus narrow C stubs. Python may be used by neither
# the runtime nor its normal validation path.
if python_files=$(rg --files --glob '*.py' 2>/dev/null); then
  printf '%s\n%s\n' 'Python files are forbidden in the LunaFlux repository:' "$python_files" >&2
  failed=1
fi

# Unowned temporary markers are rejected in code. Design documents may discuss
# the policy itself without tripping this check.
fail_matches \
  'temporary debt marker found in source or package configuration:' \
  --glob '*.mbt' --glob 'moon.pkg' --glob 'moon.mod' \
  'TODO|HACK|FIXME'

line_failure=0
while IFS= read -r source_file; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 800 ]; then
    printf '%s: %s lines; files above 800 require an ADR and split plan\n' \
      "$source_file" "$line_count" >&2
    line_failure=1
  elif [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; review cohesion before further growth\n' \
      "$source_file" "$line_count" >&2
  fi
done <<EOF
$(rg --files --glob '*.mbt' --glob '*.c' --glob '*.h' | sort)
EOF

if [ "$line_failure" -ne 0 ]; then
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux dependency and debt boundaries are valid.'
