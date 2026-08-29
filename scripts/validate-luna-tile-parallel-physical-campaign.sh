#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
runner=scripts/run-luna-tile-parallel-physical-campaign.sh
test_gate=scripts/test-luna-tile-parallel-physical-campaign.sh
fake=scripts/fixtures/luna-tile-parallel-campaign/fake-compute-sanitizer.sh

for file in "$runner" "$test_gate" "$fake"; do
  [ -f "$file" ] && [ -x "$file" ] || {
    echo "LunaTile physical campaign executable missing: $file" >&2
    exit 1
  }
  [ "$(wc -l <"$file")" -lt 500 ] || {
    echo "LunaTile physical campaign file reached 500 lines: $file" >&2
    exit 1
  }
  bash -n "$file"
done

for anchor in \
  'scripts/validate-luna-tile-parallel-cuda-probe.sh' \
  'compile_pair serial' \
  'compile_pair parallel' \
  'independent CUBIN bytes differ' \
  '--tool memcheck' \
  '--tool racecheck' \
  'parallel_candidate_sha256' \
  'manifest_bindable=false' \
  'promotion_authority=absent' \
  'lunaflux_seal_evidence_directory'; do
  grep -Fq -- "$anchor" "$runner" || {
    echo "LunaTile campaign anchor missing: $anchor" >&2
    exit 1
  }
done

if rg -ni 'nvrtc|--ptx|\.ptx|manifest_bindable=true|promotion_authority=(present|granted)' \
  "$runner" tests/luna_tile_parallel_cuda_probe >/dev/null; then
  echo 'LunaTile campaign gained JIT, PTX, manifest, or promotion authority' >&2
  exit 1
fi

scripts/validate-luna-tile-parallel-cuda-probe.sh >/dev/null
"$test_gate"
printf '%s\n' \
  'LunaTile parallel campaign remains separately compiled, physically checked, sealed, and qualification-only.'
