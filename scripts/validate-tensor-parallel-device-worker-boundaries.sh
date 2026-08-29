#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package="$root/engine/tensor_parallel_device_worker"

if rg -n 'scheduler/|service/|api/|model/llama|internal/(cuda|nccl)|LunaNexa|MoonGate' \
  "$package/moon.pkg" "$package"/*.mbt "$package"/*.mbt.md; then
  echo "tensor-parallel device worker crossed a forbidden boundary" >&2
  exit 1
fi

if rg -n 'Python|PyTorch|TVM|runtime JIT|compile(_| )at(_| )runtime' \
  "$package"/*.mbt; then
  echo "tensor-parallel device worker introduced a forbidden runtime path" >&2
  exit 1
fi

if find "$package" -name '*.mbt' -type f -exec awk 'FNR == 501 { print FILENAME }' {} + | \
  rg -n '.'; then
  echo "tensor-parallel device worker production/test file exceeds 500 lines" >&2
  exit 1
fi

if rg -n 'launch_synchronous|\.get\(|\.validate_frame\(' \
  "$package/execute.mbt"; then
  echo "tensor-parallel warmed path retained synchronous or optional capability dispatch" >&2
  exit 1
fi

if ! rg -q 'detach_frame_authentication' "$package/execute.mbt" ||
  ! rg -q 'collective_exact' "$package/execute.mbt"; then
  echo "tensor-parallel warmed path lost exact frame or collective authentication" >&2
  exit 1
fi

live_block=$(sed -n \
  '/priv struct TensorParallelLiveResources {/,/^}/p' \
  "$package/types.mbt")
if printf '%s\n' "$live_block" | rg -n '\?|Option|StagedDeviceStep|ValidatedPlanFrame'; then
  echo "tensor-parallel live resources retained optional or request capability state" >&2
  exit 1
fi

echo "tensor-parallel device-worker boundaries: ok"
