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
    'scheduler may import only canonical public model identity vocabulary:' \
    --pcre2 --glob 'scheduler/**/moon.pkg' \
    -U \
    'import\s*\{[^}]*"vectie/lunaflux/model/(?!spec"(?:,)?$)[^}]*\}(?!\s*for\s*"(?:test|wbtest)")'
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

# Canonical service frames are bounded contract codecs only. Transport,
# scheduling, tokenization, filesystem, and engine composition belong above
# this leaf package.
if [ -d service/framed_wire ]; then
  fail_matches \
    'framed service wire imports outside contracts/inference + model/spec:' \
    --pcre2 --glob 'service/framed_wire/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|model/spec")'
  fail_matches \
    'framed service wire must remain synchronous and native-ABI free:' \
    --glob 'service/framed_wire/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/framed_wire/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn RequestFrameBuffer::load\(Self, FixedArray\[Byte\], Int\)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed request decoder must retain its fixed-buffer API' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn EventFrameBuffer::load\(Self, FixedArray\[Byte\], Int\)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed event decoder must retain its fixed-buffer API' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (FramedWireLimits|RequestFrameBuffer|EventFrameBuffer|ValidatedRequestFrame|ValidatedEventFrame) \{\n  (?!// private fields)' \
      service/framed_wire/pkg.generated.mbti; then
      printf '%s\n' 'framed wire owner and limits fields must remain private' >&2
      failed=1
    fi
  fi
fi

# Incremental output owns fixed per-request decode/matcher state only. Socket,
# scheduler, worker, filesystem, and native authority stay outside this leaf.
if [ -d service/incremental_output ]; then
  fail_matches \
    'incremental output imports outside inference contracts + tokenizer:' \
    --pcre2 --glob 'service/incremental_output/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|tokenizer")'
  fail_matches \
    'incremental output must remain synchronous and native-ABI free:' \
    --glob 'service/incremental_output/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/incremental_output/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn IncrementalOutput::push_token_into\(Self, Int, FixedArray\[Byte\], destination_offset~ : Int\)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output must retain its fixed-destination token API' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct IncrementalOutput \{\n  (?!// private fields)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output state must remain opaque' >&2
      failed=1
    fi
  fi
fi

# Request admission is the synchronous tokenizer-worker bridge. It may bind
# contracts, tokenizer, monotonic time, incremental output, and the scheduler
# request value, but it must not acquire transport/process/device authority or
# become an async listener.
if [ -d service/request_admission ]; then
  fail_matches \
    'request admission imports a forbidden engine or transport owner:' \
    --glob 'service/request_admission/moon.pkg' \
    'worker_service|worker_process|internal/process|moonbitlang/async|runtime/approved_fs'
  fail_matches \
    'request admission must remain synchronous and native-ABI free:' \
    --glob 'service/request_admission/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/request_admission/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn admit\(ReceivedRequest, @tokenizer\.TokenizerSpec, @tokenizer\.TokenizerDigest, @spec\.ModelIdentity, @inference\.InferenceLimits, @monotonic_clock\.MonotonicClock\)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request admission must retain typed receipt/model/tokenizer binding' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct (RequestReceipt|AdmittedRequest) \{\n  (?!// private fields)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' 'request admission owners must remain opaque' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn receive\(@framed_wire\.RequestFrameBuffer, FixedArray\[Byte\], Int, @monotonic_clock\.MonotonicClock\) -> ReceivedRequest' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request admission must capture receipt before framed parsing' >&2
      failed=1
    fi
    if rg -q '^pub (struct RequestReceipt|fn RequestReceipt::capture)' \
      service/request_admission/pkg.generated.mbti; then
      printf '%s\n' \
        'request receipt evidence must not escape framed receipt admission' >&2
      failed=1
    fi
  fi
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

if ! rg -q '^pub fn OwnedWorkerServicePreparation::take_raw_ready\(Self\) -> WorkerService raise OwnedWorkerServicePreparationError$' \
  engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_online\(Self, @monotonic_clock.MonotonicClock\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti ||
  ! rg -q '^pub fn OwnedWorkerServicePreparation::take_prepared_online\(Self\) -> OnlineWorkerLease raise OwnedWorkerServicePreparationError$' \
    engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'owned preparation must expose exact raw and online transfers' >&2
  failed=1
fi

if rg -n 'OnlineWorkerLease::(scheduler|process|handle|request_id|request_generation|publication)' \
  engine/worker_service/pkg.generated.mbti; then
  printf '%s\n' 'online worker lease exposes raw owner or identity evidence' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  rg -n '(^|[^A-Za-z])WorkerService([^A-Za-z]|$)|OnlineWorkerLease|AdmittedRequest|TokenizedRequest|RequestHandle|SchedulerPublication|IncrementalOutput' service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'online session aggregate must not return its lower owners' >&2
  failed=1
fi

if rg -q '^  OutputFailure$' service/online_session/pkg.generated.mbti; then
  printf '%s\n' 'online output failure remains a typed aggregate error' >&2
  failed=1
fi

if [ -f service/online_session/pkg.generated.mbti ] &&
  { ! rg -q '^pub fn OnlineSession::request_cancel\(Self\) -> Unit raise OnlineSessionError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn OnlineSession::check_deadline\(Self\) -> Unit raise OnlineSessionError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^pub fn OnlineSession::progress_terminalization\(Self\) -> OnlineSessionProgress raise OnlineSessionError$' service/online_session/pkg.generated.mbti ||
    ! rg -q '^  OnlineSessionTerminalizationRequired$' service/online_session/pkg.generated.mbti; }; then
  printf '%s\n' 'online session cancellation/deadline surface drifted' >&2
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
