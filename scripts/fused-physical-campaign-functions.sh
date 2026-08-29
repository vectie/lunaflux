#!/usr/bin/env bash
# Private helpers for run-fused-parallel-physical-campaign.sh.
#
# The caller owns the campaign paths, tool identities, and fail/sha256_file
# functions. Keeping these routines separate prevents the authority-bearing
# publication flow from becoming an oversized shell unit.

lunaflux_fused_build_family() {
  local name=$1
  local source=$2
  local recipe=$3
  local root=$artifacts/builds/$name
  mkdir -p "$root/kernels" "$root/evidence"
  cp "$source" "$root/input.source.cu"
  cp "$recipe" "$root/input.recipe"
  scripts/build-luna-cuda-aot.sh "$nvcc" "$toolchain_manifest" \
    "$root/input.source.cu" "$root/input.recipe" \
    "$root/kernels" "$root/evidence" \
    >"$root/build.stdout" 2>"$root/build.stderr"
  [[ ! -s $root/build.stderr ]] || fail "$name build emitted stderr"
}

lunaflux_fused_audit_resources() {
  local family=$1 cubin=$2 source=$3 expected_shared=$4
  local root=$measurements/resources-$family
  mkdir -p "$root/rebuild"
  "$cuobjdump" --dump-sass "$cubin" >"$root/cuobjdump-sass.stdout" \
    2>"$root/cuobjdump-sass.stderr"
  "$nvdisasm" "$cubin" >"$root/nvdisasm-sass.stdout" \
    2>"$root/nvdisasm-sass.stderr"
  [[ ! -s $root/cuobjdump-sass.stderr && ! -s $root/nvdisasm-sass.stderr ]] ||
    fail "$family SASS inspection emitted stderr"
  sed -nE 's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+([A-Z][A-Z0-9_.]*).*@\2@p' \
    "$root/cuobjdump-sass.stdout" >"$root/cuobjdump-opcodes.txt"
  sed -nE 's@^[[:space:]]*/\*(0x)?[0-9A-Fa-f]+\*/[[:space:]]+([A-Z][A-Z0-9_.]*).*@\2@p' \
    "$root/nvdisasm-sass.stdout" >"$root/nvdisasm-opcodes.txt"
  local cu_count nv_count
  cu_count=$(wc -l <"$root/cuobjdump-opcodes.txt" | tr -d ' ')
  nv_count=$(wc -l <"$root/nvdisasm-opcodes.txt" | tr -d ' ')
  [[ $cu_count =~ ^[1-9][0-9]*$ && $cu_count == "$nv_count" ]] ||
    fail "$family disassemblers disagree or emitted no instructions"
  local cu_loads cu_stores nv_loads nv_stores
  cu_loads=$(grep -Ec '^LDG([.]|$)' "$root/cuobjdump-opcodes.txt")
  cu_stores=$(grep -Ec '^STG([.]|$)' "$root/cuobjdump-opcodes.txt")
  nv_loads=$(grep -Ec '^LDG([.]|$)' "$root/nvdisasm-opcodes.txt")
  nv_stores=$(grep -Ec '^STG([.]|$)' "$root/nvdisasm-opcodes.txt")
  [[ $cu_loads -gt 0 && $cu_stores -gt 0 &&
     $cu_loads == "$nv_loads" && $cu_stores == "$nv_stores" ]] ||
    fail "$family SASS lacks matching global load/store instructions"
  "$cuobjdump" --dump-resource-usage "$cubin" >"$root/cuobjdump-resources.stdout" \
    2>"$root/cuobjdump-resources.stderr"
  [[ ! -s $root/cuobjdump-resources.stderr ]] || fail "$family resource dump emitted stderr"
  cp "$source" "$root/rebuild/kernel.cu"
  (
    cd "$root/rebuild"
    TZ=UTC SOURCE_DATE_EPOCH=0 CUDA_CACHE_DISABLE=1 \
      "$nvcc" --cubin --std=c++14 --generate-code=arch=compute_120,code=sm_120 \
      -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
      --maxrregcount=128 --Werror all-warnings -Xptxas=-v kernel.cu -o kernel.cubin \
      >compiler.stdout 2>ptxas-resource.stderr
  ) || fail "$family resource rebuild failed"
  cmp -s "$cubin" "$root/rebuild/kernel.cubin" || fail "$family resource rebuild drifted"
  local registers shared stack spill_store spill_load resource_line
  [[ $(grep -Ec '^ptxas info[[:space:]]*: Used [0-9]+ registers, [0-9]+ bytes smem(,.*)?$' \
       "$root/rebuild/ptxas-resource.stderr") == 1 &&
     $(grep -Ec '^[[:space:]]*[0-9]+ bytes stack frame, [0-9]+ bytes spill stores, [0-9]+ bytes spill loads$' \
       "$root/rebuild/ptxas-resource.stderr") == 1 ]] ||
    fail "$family ptxas resource facts are ambiguous"
  read -r registers shared < <(sed -nE \
    's/^ptxas info[[:space:]]*: Used ([0-9]+) registers, ([0-9]+) bytes smem(,.*)?$/\1 \2/p' \
    "$root/rebuild/ptxas-resource.stderr")
  read -r stack spill_store spill_load < <(sed -nE \
    's/^[[:space:]]*([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads$/\1 \2 \3/p' \
    "$root/rebuild/ptxas-resource.stderr")
  [[ $(grep -Ec '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' \
       "$root/cuobjdump-resources.stdout") == 1 ]] ||
    fail "$family cuobjdump resource facts are ambiguous"
  resource_line=$(grep -E '(^|[[:space:]])REG:[0-9]+([[:space:]]|$)' \
    "$root/cuobjdump-resources.stdout")
  resource_token() {
    local prefix=$1 token value= count=0
    for token in $resource_line; do
      case $token in
        "$prefix":[0-9]*)
          value=${token#*:}
          [[ $value =~ ^[0-9]+$ ]] || fail "$family malformed $prefix resource"
          count=$((count + 1))
          ;;
      esac
    done
    [[ $count == 1 ]] || fail "$family missing or duplicate $prefix resource"
    printf '%s\n' "$value"
  }
  local cu_registers cu_shared cu_stack local_bytes
  cu_registers=$(resource_token REG)
  cu_shared=$(resource_token SHARED)
  cu_stack=$(resource_token STACK)
  local_bytes=$(resource_token LOCAL)
  [[ $registers =~ ^[1-9][0-9]*$ && $registers -le 128 &&
     $shared == "$expected_shared" && $cu_registers == "$registers" &&
     $cu_shared == "$shared" && $cu_stack == "$stack" &&
     $stack == 0 && $spill_store == 0 && $spill_load == 0 && $local_bytes == 0 ]] ||
    fail "$family resource bounds failed"
  printf -v "${family}_cu_sass_sha" '%s' "$(sha256_file "$root/cuobjdump-sass.stdout")"
  printf -v "${family}_nv_sass_sha" '%s' "$(sha256_file "$root/nvdisasm-sass.stdout")"
  printf -v "${family}_sass_count" '%s' "$cu_count"
  printf -v "${family}_global_load_count" '%s' "$cu_loads"
  printf -v "${family}_global_store_count" '%s' "$cu_stores"
  printf -v "${family}_registers" '%s' "$registers"
  printf -v "${family}_shared" '%s' "$shared"
  printf -v "${family}_local" '%s' "$local_bytes"
  printf -v "${family}_stack" '%s' "$stack"
  printf -v "${family}_spill_store" '%s' "$spill_store"
  printf -v "${family}_spill_load" '%s' "$spill_load"
}
