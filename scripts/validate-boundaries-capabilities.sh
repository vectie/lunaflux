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
  2>/dev/null | rg -v '^(internal/(approved_fs|process)|deploy/worker_executable_file)/moon.pkg$' || true)
if [ -n "$approved_fs_capability_imports" ]; then
  printf '%s\n%s\n' \
    'approved filesystem capability has unauthorized importers:' \
    "$approved_fs_capability_imports" >&2
  failed=1
fi

root_identity_calls=$(rg -l '\.require_absolute_identity\(' \
  --glob '*.mbt' 2>/dev/null | \
  rg -v '^(runtime/approved_fs|runtime/descriptor_file|runtime/instance_policy_file|tokenizer/json_file|engine/worker_process|engine/tensor_parallel_group_transport|deploy/worker_executable_file)/|^engine/tensor_parallel_rank_child/bootstrap_prepare\.mbt$' || true)
if [ -n "$root_identity_calls" ]; then
  printf '%s\n%s\n' \
    'approved root identity admission has unauthorized call sites:' \
    "$root_identity_calls" >&2
  failed=1
fi

# Offline materialization is the sole exception to final-namespace identity
# opening. One ops helper may mint the typed source->target view; only the
# mapped descriptor, policy, tokenizer, and executable loaders may borrow its
# pinned source root. Live startup continues to require exact absolute identity.
materialization_view_calls=$(rg -l '\.materialization_view\(' \
  --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null | \
  rg -v '^(runtime/approved_fs/api\.mbt|ops/runtime_instance/materialized_release_inputs\.mbt)$' || true)
if [ -n "$materialization_view_calls" ]; then
  printf '%s\n%s\n' \
    'approved materialization view has unauthorized constructors:' \
    "$materialization_view_calls" >&2
  failed=1
fi
materialization_source_calls=$(rg -l '\.source_root\(\)' \
  --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null | \
  rg -v '^(runtime/approved_fs/api\.mbt|runtime/descriptor_file/materialized_load\.mbt|runtime/instance_policy_file/load\.mbt|tokenizer/json_file/load\.mbt|deploy/worker_executable_file/admit\.mbt)$' || true)
if [ -n "$materialization_source_calls" ]; then
  printf '%s\n%s\n' \
    'approved materialization source root has unauthorized borrowers:' \
    "$materialization_source_calls" >&2
  failed=1
fi

# The rank child imports the two already-pinned inherited descriptor roles and
# authenticates each canonical source label before any descendant is opened.
# Keep both primitive calls in the one root-opening helper and make their role
# and cardinality exact; no other rank-child source receives this authority.
rank_child_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  engine/tensor_parallel_rank_child --glob '*.mbt' \
  --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null || true)
rank_child_identity_count=$(printf '%s\n' "$rank_child_identity_calls" | \
  sed '/^$/d' | wc -l | tr -d ' ')
if [ "$rank_child_identity_count" -ne 2 ] ||
  printf '%s\n' "$rank_child_identity_calls" |
    rg -v '^engine/tensor_parallel_rank_child/bootstrap_prepare\.mbt:' ||
  [ "$(rg -c 'model\.require_absolute_identity\(roots\.model_root\(\)\)' \
    engine/tensor_parallel_rank_child/bootstrap_prepare.mbt 2>/dev/null || true)" -ne 1 ] ||
  [ "$(rg -c 'kernel\.require_absolute_identity\(roots\.kernel_root\(\)\)' \
    engine/tensor_parallel_rank_child/bootstrap_prepare.mbt 2>/dev/null || true)" -ne 1 ]; then
  printf '%s\n%s\n' \
    'tensor-parallel rank child inherited-root identity binding is not exact by role:' \
    "$rank_child_identity_calls" >&2
  failed=1
fi

# The group owner performs one startup-only absolute identity check per role
# before it duplicates the pair it alone will close. Replacement uses only the
# retained pair and must not repeat or broaden this authority admission.
group_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  engine/tensor_parallel_group_transport --glob '*.mbt' \
  --glob '!*_test.mbt' --glob '!*_wbtest.mbt' 2>/dev/null || true)
group_identity_count=$(printf '%s\n' "$group_identity_calls" | \
  sed '/^$/d' | wc -l | tr -d ' ')
if [ "$group_identity_count" -ne 2 ] ||
  printf '%s\n' "$group_identity_calls" |
    rg -v '^engine/tensor_parallel_group_transport/prepare\.mbt:' ||
  [ "$(rg -c 'model_root\.require_absolute_identity\(source_roots\.model_root\(\)\)' \
    engine/tensor_parallel_group_transport/prepare.mbt 2>/dev/null || true)" -ne 1 ] ||
  [ "$(rg -c 'kernel_root\.require_absolute_identity\(source_roots\.kernel_root\(\)\)' \
    engine/tensor_parallel_group_transport/prepare.mbt 2>/dev/null || true)" -ne 1 ]; then
  printf '%s\n%s\n' \
    'tensor-parallel group root identity binding is not exact by role:' \
    "$group_identity_calls" >&2
  failed=1
fi

if ! rg -q 'lf_exec_open_no_follow\(path, status\)' \
    deploy/worker_executable_file/approved_executable.c ||
  ! rg -q 'openat\(current, component, flags\)' \
    deploy/worker_executable_file/approved_executable.c ||
  ! rg -q 'O_NOFOLLOW' deploy/worker_executable_file/approved_executable.c; then
  printf '%s\n' \
    'worker executable componentwise no-follow identity binding is incomplete' >&2
  failed=1
fi

# Descriptor admission is a second, startup-only initial-binding surface. Its
# canonical bootstrap source retains the two absolute labels, so each label
# must match the exact independently pinned capability before any descendant
# opens. Keep the primitive call private to one helper and keep the two role
# bindings explicit in each closed aggregate loader. The loader set is closed:
# legacy BF16, I8, reusable FP8, tensor parallel, and Mistral.
descriptor_identity_calls=$(rg -n '\.require_absolute_identity\(' \
  runtime/descriptor_file --glob '*.mbt' --glob '!*_test.mbt' \
  --glob '!*_wbtest.mbt' 2>/dev/null || true)
descriptor_identity_count=$(printf '%s\n' "$descriptor_identity_calls" | \
  sed '/^$/d' | wc -l | tr -d ' ')
if [ "$descriptor_identity_count" -ne 1 ] ||
  printf '%s\n' "$descriptor_identity_calls" |
    rg -v '^runtime/descriptor_file/file_owner\.mbt:'; then
  printf '%s\n%s\n' \
    'descriptor root identity primitive escaped its private binding helper:' \
    "$descriptor_identity_calls" >&2
  failed=1
fi

descriptor_binding_calls=$(rg -n 'bind_root_label\(' runtime/descriptor_file \
  --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' \
  2>/dev/null | rg -v '^runtime/descriptor_file/file_owner\.mbt:' || true)
descriptor_binding_count=$(printf '%s\n' "$descriptor_binding_calls" | \
  sed '/^$/d' | wc -l | tr -d ' ')
if [ "$descriptor_binding_count" -ne 10 ] ||
  printf '%s\n' "$descriptor_binding_calls" |
    rg -v '^runtime/descriptor_file/(load|i8_load|fp8_load|tensor_parallel_load|mistral_load)\.mbt:'; then
  printf '%s\n%s\n' \
    'descriptor roots must bind exactly once per role in aggregate admission:' \
    "$descriptor_binding_calls" >&2
  failed=1
fi

for descriptor_loader in \
  load.mbt \
  i8_load.mbt \
  fp8_load.mbt \
  tensor_parallel_load.mbt \
  mistral_load.mbt; do
  loader_binding_count=$(rg -c '^[[:space:]]*bind_root_label\($' \
    "runtime/descriptor_file/${descriptor_loader}" 2>/dev/null || true)
  if [ "$loader_binding_count" -ne 2 ]; then
    printf '%s\n' \
      "descriptor ${descriptor_loader} must contain exactly two root bindings" >&2
    failed=1
  fi
  for descriptor_role in Model Kernel; do
    role_count=$(rg -c "^[[:space:]]*Bind${descriptor_role}Root,$" \
      "runtime/descriptor_file/${descriptor_loader}" 2>/dev/null || true)
    if [ "$role_count" -ne 1 ]; then
      printf '%s\n' \
        "descriptor ${descriptor_loader} ${descriptor_role} root binding is not exact" >&2
      failed=1
    fi
  done
done

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
      rg -v '^engine/worker_process/root_bound_(binding|prepare)\.mbt$' ||
    [ "$helper_call_count" -ne 1 ] ||
    ! printf '%s\n' "$helper_calls" | \
      rg -q '^engine/worker_process/root_bound_prepare\.mbt:'; then
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


if [ "$failed" -ne 0 ]; then
  exit 1
fi
