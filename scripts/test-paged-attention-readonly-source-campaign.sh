#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
runner=$root/scripts/run-paged-attention-readonly-source-campaign.sh
scratch=$(mktemp -d /private/tmp/lunaflux-readonly-attention-source-test.XXXXXX)
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
tools=$scratch/tools
mkdir -p "$tools"

cat >"$tools/nvcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
  version=13.1.115
  [[ ${FAKE_CASE:-success} != wrong_version ]] || version=13.1.114
  if [[ ${FAKE_CASE:-success} == tool_drift ]]; then
    counter=$(dirname -- "$0")/version.counter
    value=0
    [[ ! -f $counter ]] || value=$(cat "$counter")
    value=$((value + 1))
    printf '%s' "$value" >"$counter"
    printf 'Cuda compilation tools, release 13.1, V%s.%s\n' "$version" "$value"
  else
    printf 'Cuda compilation tools, release 13.1, V%s\n' "$version"
  fi
  exit 0
fi
output=
while [[ $# -gt 0 ]]; do
  case $1 in
    -o) output=$2; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n $output ]]
cp kernel.cu "$output"
if [[ ${FAKE_CASE:-success} == nondeterministic ]]; then
  counter=$(dirname -- "$0")/build.counter
  value=0
  [[ ! -f $counter ]] || value=$(cat "$counter")
  value=$((value + 1))
  printf '%s' "$value" >"$counter"
  printf '\nnonce=%s\n' "$value" >>"$output"
fi
EOF
cat >"$tools/ptxas" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'ptxas release 13.1, V13.1.115'
EOF
chmod +x "$tools/nvcc" "$tools/ptxas"

success=$scratch/success
if ! FAKE_CASE=success "$runner" "$tools/nvcc" "$success" \
  >"$scratch/success.stdout" 2>"$scratch/success.stderr"; then
  cat "$scratch/success.stderr" >&2
  exit 1
fi
[[ ! -s $scratch/success.stderr ]]
grep -F 'outcome=paged-attention-readonly-source-campaign-published ' \
  "$scratch/success.stdout" >/dev/null
grep -Fx 'source_only=true' "$success/RESULT.txt" >/dev/null
grep -Fx 'physical_cuda_observed=false' "$success/RESULT.txt" >/dev/null
grep -Fx 'kv_cache_mutation=none' "$success/RESULT.txt" >/dev/null
grep -Fx 'manifest_bindable=false' "$success/RESULT.txt" >/dev/null
grep -Fx 'promotion_authority=absent' "$success/RESULT.txt" >/dev/null
[[ -z $(find "$success" -type f -perm -u+w -print -quit) ]]
(
  cd "$success"
  while read -r digest path; do
    [[ $(shasum -a 256 "$path" | awk '{print $1}') == "$digest" ]]
  done <OUTER_SEAL.sha256
)
if FAKE_CASE=success "$runner" "$tools/nvcc" "$success" \
  >"$scratch/overwrite.stdout" 2>"$scratch/overwrite.stderr"; then
  printf '%s\n' 'source campaign overwrote evidence' >&2
  exit 1
fi

expect_rejection() {
  local case_name=$1 output_path
  output_path=$scratch/$case_name
  rm -f "$tools/build.counter" "$tools/version.counter"
  if FAKE_CASE=$case_name "$runner" "$tools/nvcc" "$output_path" \
    >"$scratch/$case_name.stdout" 2>"$scratch/$case_name.stderr"; then
    printf 'hostile source campaign was accepted: %s\n' "$case_name" >&2
    exit 1
  fi
  [[ ! -e $output_path ]] || {
    printf 'failed source transaction published evidence: %s\n' "$case_name" >&2
    exit 1
  }
}
expect_rejection nondeterministic
expect_rejection tool_drift
expect_rejection wrong_version

printf '%s\n' 'paged-attention read-only source campaign hostile tests passed'
