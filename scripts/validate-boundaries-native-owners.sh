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

# Production foreign declarations have exactly nine narrow owners: CUDA, NCCL,
# approved descriptor-relative filesystem authority, shell-free child process
# transport, monotonic time, the online TCP buffer representation alias, and
# the inherited local drain descriptor, the separate read-once inference
# credential descriptor, and the read-once promotion-verifier-key descriptor,
# each under a dedicated internal ABI package.
# Positive-controlled allocation harnesses, the exact approved-root child /
# parent E2E probes, and the promotion-verifier-key fixed-descriptor E2E probe
# are the sole exceptions. Their narrow C shims inspect process state or
# generated allocation entry points and are not imported by a production
# package.
fail_matches \
  'production native declarations are only allowed under approved internal ABI packages:' \
  --glob '*.mbt' --glob '!internal/cuda/**' \
  --glob '!internal/inherited_drain/**' \
  --glob '!internal/inference_credential/**' \
  --glob '!internal/nccl/**' \
  --glob '!internal/approved_fs/**' \
  --glob '!internal/monotonic_clock/**' \
  --glob '!internal/online_tcp_buffer_alias/**' \
  --glob '!internal/process/**' \
  --glob '!internal/promotion_verifier_key/**' \
  --glob '!deploy/worker_executable_file/**' \
  --glob '!tests/worker_executable_fixture/**' \
  --glob '!tests/hot_path_alloc/**' \
  --glob '!tests/rank_group_wire_alloc/**' \
  --glob '!tests/rank_child_control_alloc/**' \
  --glob '!tests/device_step_alloc/**' \
  --glob '!tests/device_worker_alloc/**' \
  --glob '!tests/tensor_parallel_device_worker_alloc/**' \
  --glob '!tests/inherited_drain_e2e/**' \
  --glob '!tests/inference_credential_e2e/**' \
  --glob '!tests/promotion_verifier_key_e2e/**' \
  --glob '!cmd/approved_root_echo/**' \
  --glob '!tests/approved_root_inheritance_e2e/**' \
  'extern\s+"[cC]"|#external'

expected_internal_abi_owners="$(cat <<'EOF'
internal/approved_fs
internal/cuda
internal/inference_credential
internal/inherited_drain
internal/monotonic_clock
internal/nccl
internal/online_tcp_buffer_alias
internal/process
internal/promotion_verifier_key
EOF
)"
actual_internal_abi_owners="$(rg -l 'extern\s+"[cC]"|#external' internal \
  --glob '*.mbt' | sed -E 's#^(internal/[^/]+).*#\1#' | sort -u)"
if [ "$actual_internal_abi_owners" != "$expected_internal_abi_owners" ]; then
  printf '%s\n' 'production internal ABI owner set drifted from exactly nine' >&2
  failed=1
fi


# Exact online-TCP ABI and public-surface closure is owned by the service
# boundary driver. Keeping that single allowlist avoids parallel policy drift.

if [ "$failed" -ne 0 ]; then
  exit 1
fi
