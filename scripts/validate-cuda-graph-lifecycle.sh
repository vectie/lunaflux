#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

for symbol in cuStreamBeginCapture cuStreamEndCapture \
  cuGraphInstantiateWithFlags cuGraphDestroy cuGraphExecDestroy cuGraphLaunch; do
  if ! rg -Fq "lf_symbol(api->driver_library, \"$symbol\")" \
    internal/cuda/loader.c; then
    printf 'optional CUDA graph symbol is not admitted: %s\n' "$symbol" >&2
    exit 1
  fi
  if rg -Fq "LF_LOAD_REQUIRED($symbol" internal/cuda/loader.c; then
    printf 'CUDA graph support must remain optional: %s\n' "$symbol" >&2
    exit 1
  fi
done

if rg -n 'cuGraphExec(Update|KernelNodeSetParams)|cuGraphAdd' \
  internal/cuda device engine/device_step 2>/dev/null; then
  echo 'live graph mutation or node synthesis escaped the fixed capture seam' >&2
  exit 1
fi

launch_body=$(sed -n \
  '/^int32_t lf_ordered_graph_launch(/,/^}/p' \
  internal/cuda/ordered_graph.c)
if [ -z "$launch_body" ]; then
  echo 'captured graph launch implementation is missing' >&2
  exit 1
fi
if matches=$(printf '%s\n' "$launch_body" | rg -n \
  'moonbit_make|moonbit_incref|moonbit_decref|malloc|calloc|realloc|free|cuStreamBeginCapture|cuStreamEndCapture' \
  2>/dev/null); then
  printf '%s\n%s\n' 'warmed captured graph launch allocates or captures:' "$matches" >&2
  exit 1
fi

if ! rg -q 'policy == LF_ORDERED_CAPTURE_WITH_EAGER_FALLBACK' \
  internal/cuda/ordered_graph.c ||
  ! rg -q 'PagedCapturedWithEagerFallback => OrderedKernelCaptureWithEagerFallback' \
    engine/device_step/paged_ordered_executor_prepare.mbt; then
  echo 'eager fallback is not guarded by the explicit admitted policy' >&2
  exit 1
fi

if ! rg -q 'OrderedKernelCaptured =>' engine/device_step/paged_executor_run.mbt ||
  ! rg -q 'ordered.launch_captured()' engine/device_step/paged_executor_run.mbt; then
  echo 'paged execution does not dispatch the startup-selected captured graph' >&2
  exit 1
fi

echo 'LunaFlux startup-only CUDA graph lifecycle boundary passed.'
