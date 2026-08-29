#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  printf '%s\n' "Phase-7 physical topology diagnostic gate failed: $1" >&2
  exit 1
}

package=ops/physical_topology_diagnostic
command_package=cmd/phase7_topology_diagnostic
capture_script=scripts/run-phase7-physical-topology-diagnostic.sh

if rg -n \
  'vectie/lunaflux/(engine/(worker|rank_group|tensor_parallel_group)|internal/(cuda|nccl|process)|scheduler|service|runtime/approved_fs)' \
  "$package/moon.pkg" "$command_package/moon.pkg"; then
  fail 'diagnostic acquired execution, process, scheduler, or filesystem authority'
fi

if rg -n \
  'open_context\(|create_(tensor_collective|communicator|stream)\(|allocate\(|enable_peer\(|\b(spawn|fork|exec)\(|ApprovedRoot|nvidia-smi|extern "c"|@nccl\.|@cuda\.' \
  "$package" "$command_package" --glob '*.mbt'; then
  fail 'diagnostic source contains a forbidden live-authority operation'
fi

run_body=$(sed -n '/^pub fn run_local(/,/^}/p' "$package/run.mbt")
for required in \
  'observe_collective_runtime()' \
  '@device.probe()' \
  '@device.can_access_peer(' \
  'evaluate_observations('; do
  printf '%s\n' "$run_body" | rg -Fq "$required" ||
    fail "physical runner lost $required"
done
collective_body=$(sed -n \
  '/^fn observe_collective_runtime(/,/^}/p' "$package/run.mbt")
for required in \
  '@device.probe_tensor_collective_runtime()' \
  '@device.admit_tensor_collective_runtime('; do
  printf '%s\n' "$collective_body" | rg -Fq "$required" ||
    fail "collective authentication lost $required"
done

for required in \
  'schema: lunaflux.phase7.physical_topology.v1' \
  'outcome: rejected' \
  'device.0.name_utf8_hex: 736d313230' \
  'peer.0.accessible: false' \
  'peer.1.accessible: false' \
  'collective.availability: library_missing' \
  'rejection.reason: heterogeneous_target' \
  'authority.context_opened: false' \
  'authority.device_allocation_created: false' \
  'authority.communicator_created: false' \
  'authority.rank_process_spawned: false'; do
  rg -Fq "$required" "$package/diagnostic_test.mbt" ||
    fail "mixed-host rejection test lost $required"
done

if ! rg -q -U \
  'pub struct PhysicalTopologyDiagnosticEvidence \{\n  priv driver_version' \
  "$package/types.mbt"; then
  fail 'evidence fields are no longer immutable and opaque'
fi

mbti="$package/pkg.generated.mbti"
for signature in \
  'pub fn evaluate_observations(driver_version~ : Int, ArrayView[PhysicalTopologyDeviceObservation], ArrayView[PhysicalPeerObservation], collective~ : PhysicalCollectiveObservation) -> PhysicalTopologyDiagnosticEvidence raise PhysicalTopologyDiagnosticError' \
  'pub fn render(PhysicalTopologyDiagnosticEvidence) -> String' \
  'pub fn run_local() -> PhysicalTopologyDiagnosticEvidence'; do
  rg -Fqx "$signature" "$mbti" || fail "generated API drifted: $signature"
done

device_mbti=device/pkg.generated.mbti
for signature in \
  'pub fn probe_tensor_collective_runtime() -> TensorCollectiveRuntimeInventory' \
  'pub fn TensorCollectiveRuntimeInventory::availability(Self) -> TensorCollectiveRuntimeAvailability' \
  'pub fn TensorCollectiveRuntimeInventory::version_code(Self) -> Int'; do
  rg -Fqx "$signature" "$device_mbti" ||
    fail "device collective inventory API drifted: $signature"
done

main_body=$(sed -n '/^fn main/,/^}/p' "$command_package/main.mbt")
printf '%s\n' "$main_body" | rg -Fq '@physical_topology_diagnostic.run_local()' ||
  fail 'command does not execute the product-owned runner'
if printf '%s\n' "$main_body" | rg -n 'get_cli_args|exit|abort|raise'; then
  fail 'unsupported diagnostic outcome can no longer exit successfully'
fi

for required in \
  'moon build cmd/phase7_topology_diagnostic' \
  '_build/native/debug/build/cmd/phase7_topology_diagnostic/phase7_topology_diagnostic.exe' \
  '"$runner" >"$evidence_dir/stdout.log" 2>"$evidence_dir/stderr.log"' \
  'evidence_schema=lunaflux.phase7.physical_topology.capture.v1' \
  'runner_sha256=' \
  'chmod 0444'; do
  rg -Fq "$required" "$capture_script" ||
    fail "immutable capture lost $required"
done
if rg -n 'moon run|nvidia-smi|tee ' "$capture_script"; then
  fail 'immutable capture no longer runs only the prebuilt product command'
fi

for file in $(find "$package" "$command_package" -type f \
  \( -name '*.mbt' -o -name '*.mbt.md' \) | sort); do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$file exceeds the 500-line budget ($lines)"
done

scripts/validate-device-topology-boundaries.sh
scripts/validate-tensor-parallel-collective-boundaries.sh
scripts/validate-nccl-abi.sh
scripts/validate-cuda-peer-access-sanitizer.sh
scripts/validate-nccl-sanitizer.sh

printf '%s\n' 'LunaFlux Phase-7 physical topology diagnostic boundary is valid.'
