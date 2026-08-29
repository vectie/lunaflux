#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
validator="$repo_root/scripts/validate-debt-policy.sh"
fixture_root=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-debt-policy-test.XXXXXX")
trap 'rm -rf "$fixture_root"' EXIT HUP INT TERM

temporary_marker='TO''DO'
shortcut_marker='H''ACK'
fix_marker='FIX''ME'

reset_fixture() {
  rm -rf "$fixture_root/case"
  mkdir -p "$fixture_root/case/engine/feature"
}

expect_pass() {
  name=$1
  if ! "$validator" "$fixture_root/case" >/dev/null 2>&1; then
    printf 'expected pass: %s\n' "$name" >&2
    exit 1
  fi
}

expect_fail() {
  name=$1
  if "$validator" "$fixture_root/case" >/dev/null 2>&1; then
    printf 'expected failure: %s\n' "$name" >&2
    exit 1
  fi
}

reset_fixture
cat > "$fixture_root/case/engine/feature/owned.mbt" <<EOF
/// ${temporary_marker}(owner=phase-9/debt; remove_when=the replacement catalog ships; latest_phase=phase-9)
pub fn answer() -> Int {
  let mut value = 41
  value += 1
  value
}
EOF
printf '%s\n' \
  "# ${temporary_marker}(owner=phase-9/debt; remove_when=the replacement catalog ships; latest_phase=phase-9)" \
  > "$fixture_root/case/engine/feature/owned.sh"
mkdir -p "$fixture_root/case/_build" "$fixture_root/case/.repos" \
  "$fixture_root/case/node_modules" "$fixture_root/case/vendor"
printf '// %s ignored build output\n' "$temporary_marker" \
  > "$fixture_root/case/_build/generated.mbt"
printf '// %s ignored dependency\n' "$fix_marker" \
  > "$fixture_root/case/vendor/dependency.mbt"
printf '// %s ignored fetched CUDA\n' "$shortcut_marker" \
  > "$fixture_root/case/.repos/generated.cu"
printf '# %s ignored package script\n' "$temporary_marker" \
  > "$fixture_root/case/node_modules/dependency.sh"
printf '// %s allowed outside a release-gated package\n' "$shortcut_marker" \
  > "$fixture_root/case/engine/feature/cohesive_helpers.mbt"
expect_pass 'owned temporary marker, local mutation, and excluded trees'

for source_name in debt.mbt debt.c debt.h debt.cc debt.cpp debt.cu debt.cuh debt.sh moon.pkg moon.mod; do
  reset_fixture
  printf '// %s later\n' "$temporary_marker" \
    > "$fixture_root/case/engine/feature/$source_name"
  expect_fail "unowned temporary marker in $source_name"
done

reset_fixture
printf '// %s(owner=phase-9/debt; remove_when=after replacement)\n' \
  "$temporary_marker" > "$fixture_root/case/engine/feature/debt.mbt"
expect_fail 'temporary marker missing latest phase'

reset_fixture
printf 'let note = "%s later"\n' "$temporary_marker" \
  > "$fixture_root/case/engine/feature/reserved_literal.mbt"
expect_fail 'temporary marker spelling is reserved inside literals'

reset_fixture
mkdir -p "$fixture_root/case/kernels/fixture"
printf '// %s temporary shortcut\n' "$shortcut_marker" \
  > "$fixture_root/case/kernels/fixture/probe.cu"
expect_fail 'hot-path CUDA shortcut marker'

for abi_owner in approved_fs cuda monotonic_clock nccl online_tcp_buffer_alias process; do
  reset_fixture
  mkdir -p "$fixture_root/case/internal/$abi_owner"
  printf '// %s native ownership\n' "$fix_marker" \
    > "$fixture_root/case/internal/$abi_owner/bridge.c"
  expect_fail "native ABI release marker in internal/$abi_owner"
done

reset_fixture
printf '%s\n' 'let mut runtime = 0' \
  > "$fixture_root/case/engine/feature/global.mbt"
expect_fail 'global mutable runtime declaration'

reset_fixture
printf '%s\n' 'pub fn helper() -> Unit { () }' \
  > "$fixture_root/case/engine/feature/utils.mbt"
expect_fail 'vague dumping-ground filename'

reset_fixture
line=1
while [ "$line" -lt 500 ]; do
  printf '%s\n' '/// bounded line' >> \
    "$fixture_root/case/engine/feature/unterminated.mbt"
  line=$((line + 1))
done
printf '%s' '/// unterminated final logical line' >> \
  "$fixture_root/case/engine/feature/unterminated.mbt"
expect_fail '500 logical MoonBit lines with unterminated final line'

reset_fixture
line=0
while [ "$line" -lt 800 ]; do
  printf '%s\n' '# bounded shell line' >> \
    "$fixture_root/case/engine/feature/bounded.sh"
  line=$((line + 1))
done
expect_pass '800-line generic first-party source'
printf '%s\n' '# line beyond the hard ceiling' >> \
  "$fixture_root/case/engine/feature/bounded.sh"
expect_fail '801-line generic first-party source'

printf '%s\n' 'LunaFlux technical-debt policy self-tests passed.'
