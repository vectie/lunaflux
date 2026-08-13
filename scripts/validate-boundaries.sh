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

# Production foreign declarations have exactly three narrow owners: CUDA,
# approved descriptor-relative filesystem authority, and shell-free child
# process transport, each under its dedicated internal ABI package.
# The two positive-controlled release allocation harnesses are the sole test
# exceptions: their narrow C shims instrument generated MoonBit allocation
# entry points and are never imported by a production package.
fail_matches \
  'production native declarations are only allowed under approved internal ABI packages:' \
  --glob '*.mbt' --glob '!internal/cuda/**' \
  --glob '!internal/approved_fs/**' \
  --glob '!internal/process/**' \
  --glob '!tests/hot_path_alloc/**' \
  --glob '!tests/device_step_alloc/**' \
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
