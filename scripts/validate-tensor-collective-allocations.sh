#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test device --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/device/device.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'tensor-collective release allocation evidence is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ { candidate = 1; body = $0 ORS; next }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1; depth = 1; printf "%s", body; candidate = 0; next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$generated_c"
}

allocation_pattern='moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string'
unexpected_allocations() {
  rg -n "$allocation_pattern" |
    rg -v 'moonbit_malloc.*DeviceError_2e' || true
}
authenticate_body="$(extract_definition 'authenticate__tensor__collective__claim(')"
query_body="$(extract_definition 'require__tensor__collective__query__tokens(')"
elements_body="$(extract_definition 'checked__tensor__collective__elements(')"
if [ -z "$authenticate_body" ] || [ -z "$query_body" ] ||
  [ -z "$elements_body" ]; then
  printf '%s\n' 'tensor-collective warmed production call edge disappeared' >&2
  exit 1
fi
submit_source="$(sed -n \
  '/^pub fn TensorCollectiveOwner::submit(/,/^\/\/\/|$/p' \
  device/tensor_collective_execute.mbt)"
if [ -z "$submit_source" ] ||
  ! printf '%s\n' "$submit_source" |
    rg -F -q 'authenticate_tensor_collective_claim('; then
  printf '%s\n' 'collective submit no longer reaches scalar authentication' >&2
  exit 1
fi
if ! printf '%s\n' "$submit_source" |
  rg -F -q 'require_tensor_collective_query_tokens('; then
  printf '%s\n' 'collective submit no longer authenticates query tokens' >&2
  exit 1
fi
if ! printf '%s\n' "$submit_source" |
  rg -F -q 'checked_tensor_collective_elements('; then
  printf '%s\n' 'collective submit no longer derives checked element counts' >&2
  exit 1
fi
if [ "$(printf '%s\n' "$submit_source" | rg -c '\.collective_token\(\)')" -ne 4 ]; then
  printf '%s\n' \
    'collective submit does not convert exactly four opaque device tokens' >&2
  exit 1
fi
token_body="$(sed -n \
  '/^lf_device_context_token \*lunaflux_device_interop_context_token(/,/^}/p; /^lf_device_region_token \*lunaflux_device_interop_region_token(/,/^}/p; /^lf_device_queue_token \*lunaflux_device_interop_queue_token(/,/^}/p' \
  internal/cuda/collective_interop.c)"
native_submit_body="$(sed -n \
  '/^int32_t lf_nccl_communicator_submit_bf16(/,/^}/p' \
  internal/nccl/collectives.c)"
if [ -z "$token_body" ] || [ -z "$native_submit_body" ]; then
  printf '%s\n' 'opaque collective native call edge disappeared' >&2
  exit 1
fi
if printf '%s\n%s\n' "$token_body" "$native_submit_body" |
  rg -n 'moonbit_make|malloc|calloc|realloc|free' 2>/dev/null; then
  printf '%s\n' 'opaque collective token/submit edge allocates' >&2
  exit 1
fi
if [ -n "$(printf '%s\n' "$authenticate_body" | unexpected_allocations)" ]; then
  printf '%s\n' 'tensor-collective warmed scalar authentication allocates' >&2
  exit 1
fi
if [ -n "$(printf '%s\n' "$query_body$elements_body" |
  unexpected_allocations)" ]; then
  printf '%s\n' 'tensor-collective scalar preparation allocates' >&2
  exit 1
fi
if ! rg -q 'Bytes4make\([^,]+, 77\)' "$generated_c"; then
  printf '%s\n' 'tensor-collective allocation positive control is ineffective' >&2
  exit 1
fi
if rg -n 'TensorCollectiveLaunchClaim' device/pkg.generated.mbti \
  device/tensor_collective_*.mbt; then
  printf '%s\n' 'heap-shaped launch claim remains in the device surface' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux tensor-collective warmed scalar allocation gate passed.'
