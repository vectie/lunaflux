#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunatile-campaign-test.XXXXXX")
scratch=$(CDPATH= cd -- "$scratch" && pwd -P)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM

fail() { printf 'LunaTile campaign test failed: %s\n' "$1" >&2; exit 1; }
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

mkdir "$scratch/tools"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-nvcc.sh" "$scratch/tools/nvcc"
cp "$root/scripts/fixtures/luna-cuda-aot/fake-ptxas.sh" "$scratch/tools/ptxas"
cp "$root/scripts/fixtures/luna-tile-parallel-campaign/fake-compute-sanitizer.sh" \
  "$scratch/tools/compute-sanitizer"
chmod 0555 "$scratch/tools/nvcc" "$scratch/tools/ptxas" \
  "$scratch/tools/compute-sanitizer"
runner=$root/scripts/run-luna-tile-parallel-physical-campaign.sh

success=$scratch/success
"$runner" "$scratch/tools/nvcc" "$scratch/tools/compute-sanitizer" \
  "$success" 1 >"$scratch/success.stdout" 2>"$scratch/success.stderr" || {
  sed -n '1,160p' "$scratch/success.stderr" >&2
  fail 'ordinary fake campaign was rejected'
}
[[ ! -s $scratch/success.stderr ]] || fail 'success emitted stderr'
[[ -d $success && ! -L $success && ! -w $success ]] ||
  fail 'success evidence is absent or writable'
grep -F 'outcome=lunatile-parallel-physical-campaign-published' \
  "$scratch/success.stdout" >/dev/null || fail 'publication outcome missing'
grep -Fx 'outcome=lunatile-parallel-physical-campaign-pass' \
  "$success/CAMPAIGN_RESULT.txt" >/dev/null || fail 'campaign result missing'
grep -F '  CAMPAIGN_RESULT.txt' "$success/FILES.sha256" >/dev/null ||
  fail 'campaign result is outside manifest'
while read -r digest relative; do
  relative=${relative#  }
  [[ $(sha256_file "$success/$relative") == "$digest" ]] ||
    fail "sealed file drifted: $relative"
done <"$success/FILES.sha256"

if "$runner" "$scratch/tools/nvcc" "$scratch/tools/compute-sanitizer" \
  "$success" 1 >/dev/null 2>&1; then
  fail 'runner overwrote existing evidence'
fi

assert_rejected() {
  local mode=$1 output=$scratch/rejected-$1
  if FAKE_LUNATILE_CAMPAIGN_MODE=$mode \
    "$runner" "$scratch/tools/nvcc" "$scratch/tools/compute-sanitizer" \
    "$output" 1 >/dev/null 2>&1; then
    fail "hostile campaign passed: $mode"
  fi
  [[ ! -e $output && ! -L $output ]] || fail "hostile output published: $mode"
  compgen -G "$scratch/.rejected-$mode.partial.*" >/dev/null &&
    fail "hostile staging retained: $mode"
  return 0
}

assert_rejected dirty-sanitizer
assert_rejected extra-summary
assert_rejected stderr
assert_rejected identity-mismatch

if FAKE_LUNA_CUDA_NONDETERMINISTIC=1 \
  "$runner" "$scratch/tools/nvcc" "$scratch/tools/compute-sanitizer" \
  "$scratch/rejected-nondeterministic" 1 >/dev/null 2>&1; then
  fail 'nondeterministic CUBIN campaign passed'
fi
[[ ! -e $scratch/rejected-nondeterministic ]] ||
  fail 'nondeterministic campaign published evidence'

printf '%s\n' 'LunaTile parallel physical campaign hostile transaction passed.'
