#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 ABSOLUTE_NVCC_13_1" >&2
  exit 2
fi
nvcc=$1
case "$nvcc" in /*) ;; *) echo 'nvcc path must be absolute' >&2; exit 2;; esac
[ -x "$nvcc" ] || { echo 'nvcc is not executable' >&2; exit 2; }

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
scripts/validate-approved-bf16-model-physical.sh >/dev/null

work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-approved-bf16-model.XXXXXX")
cleanup() {
  chmod -R u+w "$work" 2>/dev/null || true
  rm -rf -- "$work"
}
trap cleanup EXIT HUP INT TERM

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" >"$work/driver.txt"
driver_identity=$(sed -n '6s/^driver_identity_sha256=//p' "$work/driver.txt")
compiler_version=$(sed -n '7s/^compiler_version=//p' "$work/driver.txt")
[ "$compiler_version" = 13.1.115 ] || {
  echo "approved-model probe requires CUDA compiler 13.1.115" >&2
  exit 1
}
compiler_major=${compiler_version%%.*}
compiler_tail=${compiler_version#*.}
compiler_minor=${compiler_tail%%.*}
compiler_patch=${compiler_tail#*.}

toolchain_manifest=$work/toolchain.v1
{
  printf '%s\n' 'schema=lunaflux-approved-complete-cuda-toolchain-v1'
  printf 'driver_identity_sha256=%s\n' "$driver_identity"
} >"$toolchain_manifest"
toolchain_sha=$(sha256_file "$toolchain_manifest")

if ! moon build --target native tests/approved_bf16_model_physical --deny-warn \
  >"$work/build.stdout" 2>"$work/build.stderr"; then
  echo 'MoonBit approved-model probe build failed' >&2
  sed -n '1,120p' "$work/build.stdout" >&2
  sed -n '1,120p' "$work/build.stderr" >&2
  exit 1
fi
probe=$repo_root/_build/native/debug/build/tests/approved_bf16_model_physical/approved_bf16_model_physical.exe
[ -x "$probe" ] || {
  echo 'MoonBit approved-model probe executable is missing' >&2
  exit 1
}

model_root=$repo_root/tests/reference_corpus
export_root=$work/export
"$probe" export "$model_root" "$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" "$export_root" \
  >"$work/export.stdout" 2>"$work/export.stderr"
[ ! -s "$work/export.stderr" ] || {
  echo 'candidate export emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/export.stderr" >&2
  exit 1
}
grep -Eq '^outcome=approved-bf16-candidates-exported target=sm_120 operations=21 candidate_set_sha256=[0-9a-f]{64} inventory_sha256=[0-9a-f]{64} export_record_sha256=[0-9a-f]{64}$' \
  "$work/export.stdout"

candidate_root=$export_root/candidate-root
candidate_inventory=$export_root/candidate.files.sha256
candidate_inventory_sha=$(sha256_file "$candidate_inventory")
compiled_root=$work/compiled
scripts/build-luna-bf16-kernel-set.sh \
  "$nvcc" "$toolchain_manifest#sha256=$toolchain_sha" \
  "$candidate_root" "$candidate_inventory#sha256=$candidate_inventory_sha" \
  "$compiled_root" >"$work/compiler.stdout" 2>"$work/compiler.stderr"
[ ! -s "$work/compiler.stderr" ] || {
  echo 'offline compiler emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/compiler.stderr" >&2
  exit 1
}

scripts/verify-luna-bf16-kernel-set.sh "$compiled_root" \
  >"$work/verifier.stdout" 2>"$work/verifier.stderr"
[ ! -s "$work/verifier.stderr" ] || {
  echo 'compiled-set verifier emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/verifier.stderr" >&2
  exit 1
}

runtime_root=$work/runtime
mkdir "$runtime_root"
cp -R "$compiled_root/sha256" "$runtime_root/sha256"
[ "$(find "$runtime_root/sha256" -type f -name '*.cubin' | wc -l | tr -d ' ')" -ge 1 ] || {
  echo 'runtime module root is empty' >&2
  exit 1
}

"$probe" run "$model_root" "$compiled_root" "$runtime_root" \
  "$toolchain_manifest" "$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" \
  >"$work/runtime.stdout" 2>"$work/runtime.stderr"
[ ! -s "$work/runtime.stderr" ] || {
  echo 'approved-model runtime emitted unexpected stderr' >&2
  sed -n '1,120p' "$work/runtime.stderr" >&2
  exit 1
}
grep -Eq '^outcome=approved-bf16-model-sm120-pass revision=3579d71fd57e04f5a364d824d3a2ec3e913dbb67 plan_executed=1 selected_logits=11 max_abs_error=[0-9.eE+-]+ greedy_tokens=1031,2185,688,2844 same_page_kv=pass capture=required artifacts=authenticated weights=authenticated resources=closed scope=tiny-approved-model-numerics$' \
  "$work/runtime.stdout"
cat "$work/runtime.stdout"
