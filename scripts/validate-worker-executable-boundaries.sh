#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

release_target=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-worker-exec-release.XXXXXX")
cleanup() { rm -rf "$release_target"; }
trap cleanup EXIT HUP INT TERM

if rg -n \
  'FixtureExecutableAuthority|WorkerExecutableFixtureAuthority|verify_fixture_authority|admit_unchecked|prepare_child_fixture|prepare_owned_fixture|prepare_with_fixture_executable|prepare_owned_luna_.*_fixture' \
  . --glob '*.mbt' --glob '*.mbti' --glob '!internal/process/*_wbtest.mbt'; then
  fail 'production/generated APIs still expose raw executable fixture authority'
fi

for manifest in $(find . \
  -path './tests' -prune -o \
  -path './_build' -prune -o \
  -path './.git' -prune -o \
  -name moon.pkg -print); do
  if ! awk '
    /^import \{/ { in_import = 1; has_test_dependency = 0 }
    in_import && /vectie\/lunaflux\/tests\/(process_native_fixture|worker_executable_native_probe|worker_executable_fixture)/ {
      has_test_dependency = 1
    }
    in_import && /^}/ {
      if (has_test_dependency && $0 !~ /for "(test|wbtest)"/) exit 1
      in_import = 0
    }
  ' "$manifest"; then
    fail "production import block may not link a worker executable test package: $manifest"
  fi
done

if rg -n 'b"/bin/cat"' \
    engine/worker_process \
    engine/rank_group_process \
    engine/tensor_parallel_group_transport ||
  ! rg -U -q \
    '#if defined\(__linux__\)[\s\S]*"/usr/bin/cat"[\s\S]*#else[\s\S]*"/bin/cat"' \
    tests/worker_executable_fixture/read_fixture.c; then
  fail 'descriptor-execution tests must use the platform exact cat path'
fi

if rg -n 'fixture|test_(link|close)' \
    deploy/worker_executable_file/approved_executable.c \
    internal/approved_fs_capability/approved_fs_capability.h \
    internal/process/process.c \
    internal/process/process_approved_spawn.c; then
  fail 'production native executable authority still contains a test seam'
fi

if [ -e internal/approved_fs_capability/approved_executable_capability.c ] ||
  ! rg -q 'static inline int32_t lf_approved_executable_duplicate' \
    internal/approved_fs_capability/approved_fs_capability.h; then
  fail 'executable duplication must remain header-owned without a link archive'
fi

if rg -n \
  'raw_spawn_prepared\(|raw_spawn_prepared_with_approved_roots' \
  internal/process/ffi.mbt internal/process/pkg.generated.mbti; then
  fail 'production process API still exposes pathname spawn'
fi

if ! rg -q \
    'priv approved_executable : @executable_capability.ApprovedExecutableHandle$' \
    internal/process/types.mbt ||
  rg -q 'priv approved_executable : .*\?' internal/process/types.mbt ||
  ! rg -q \
    'priv approved_executable : @worker_executable_file.WorkerExecutableAdmission$' \
    engine/worker_process/root_bound_types.mbt; then
  fail 'live executable authority admits an impossible optional state'
fi

if ! rg -q 'MFD_CLOEXEC \| MFD_ALLOW_SEALING' \
    deploy/worker_executable_file/approved_executable.c ||
  ! rg -q \
    'F_SEAL_WRITE \| F_SEAL_GROW \| F_SEAL_SHRINK \| F_SEAL_SEAL' \
    deploy/worker_executable_file/approved_executable.c ||
  ! rg -U -q \
    'int prior = owner->fd;[\s\S]*lf_exec_close\(prior\)[\s\S]*owner->fd = sealed;' \
    deploy/worker_executable_file/approved_executable.c; then
  fail 'Linux executable snapshot is not sealed before safe authority publish'
fi

if ! rg -q 'fexecve\(5, argv, sanitized_environment\)' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'SYS_close_range, 6u, UINT_MAX, 0u' \
    internal/process/process_approved_spawn.c ||
  ! rg -q 'close\(2\)' internal/process/process_approved_spawn.c ||
  ! rg -U -q 'pthread_sigmask\(SIG_BLOCK,[\s\S]*fork\(\)' \
    internal/process/process_approved_spawn.c ||
  ! rg -U -q 'signal_number < NSIG[\s\S]*sigprocmask\(SIG_SETMASK' \
    internal/process/process_approved_spawn.c; then
  fail 'approved spawn lost descriptor or inherited-signal normalization'
fi


# A fresh isolated build includes skipped Linux descriptor-execution tests and
# links every test executable without running platform-specific child routes.
moon test --target native --release --build-only --include-skipped \
  --deny-warn --target-dir "$release_target" >/dev/null
release_root="$release_target/native/release/test"
process_archive=$(find "$release_root" \
  -path '*/internal/process/libprocess*.a' -print -quit)
executable_archive=$(find "$release_root" \
  -path '*/deploy/worker_executable_file/libworker_executable_file*.a' -print -quit)
if [ -z "$process_archive" ] || [ -z "$executable_archive" ]; then
  fail 'release executable-authority archives were not produced'
fi
if nm -g "$process_archive" "$executable_archive" | rg -q \
    'lunaflux_process_spawn_prepared$|lunaflux_process_spawn_prepared_with_approved_roots$|lf_process_spawn_path|lf_approved_executable_duplicate|lunaflux_process_test_spawn_fixture|lunaflux_worker_executable_.*(fixture|test)'; then
  fail 'release archives export a pathname or test executable seam'
fi
if find "$release_root" \
    -path '*/tests/*' -prune -o \
    -type f \( -name '*.a' -o -name '*.o' \) -print \
    -exec nm -g {} + 2>/dev/null | rg -q \
    'lunaflux_process_test_spawn_fixture|lunaflux_worker_executable_.*(fixture|test)'; then
  fail 'release graph linked a worker executable test-native object'
fi

if ! rg -q 'inputs.admitted_rank_child_activation_path != executable.activation_path\(\)' \
    engine/tensor_parallel_group_transport/prepare.mbt ||
  ! rg -q 'inputs.rank_child_executable_digest_sha256 != executable.digest\(\).as_hex\(\)' \
    engine/tensor_parallel_group_transport/prepare.mbt; then
  fail 'tensor-parallel planning evidence is not joined to live authority'
fi

if ! rg -q 'pub struct MaterializedWorkerExecutableEvidence' \
    deploy/worker_executable_file/types.mbt ||
  rg -q \
    'MaterializedWorkerExecutableEvidence.*prepare_child|prepare_child.*MaterializedWorkerExecutableEvidence' \
    deploy/worker_executable_file/pkg.generated.mbti; then
  fail 'root-free evidence can be confused with live activation authority'
fi

for source_file in \
  deploy/worker_executable_file/approved_executable.c \
  internal/process/process_approved_spawn.c \
  engine/worker_process/root_bound.mbt \
  engine/worker_service/owned_prepare.mbt; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; executable authority files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux worker executable authority boundary is valid.'
