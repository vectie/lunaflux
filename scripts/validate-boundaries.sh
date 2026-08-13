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
    '^pub fn new\(@core\.Scheduler, WorkerServiceBinding, @worker_process\.RootBoundWorkerProcessSupervisor\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' \
      'worker service must own the root-bound process supervisor' >&2
    failed=1
  fi
  if ! rg -q \
    '^pub fn WorkerService::restart\(Self\)' \
    engine/worker_service/pkg.generated.mbti; then
    printf '%s\n' 'worker service restart must remain zero-argument' >&2
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
  --glob '!cmd/approved_root_echo/**' \
  --glob '!tests/approved_root_inheritance_e2e/**' \
  'extern\s+"[cC]"|#external'

# Internal ABI concrete types must never become part of a public package
# interface. Generated interfaces are authoritative for this boundary.
fail_matches \
  'public package interface leaks an internal ABI type:' \
  --glob 'pkg.generated.mbti' --glob '!internal/**' \
  'vectie/lunaflux/internal/'

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
