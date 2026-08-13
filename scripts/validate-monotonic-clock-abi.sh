#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

extern_count=$(rg -n 'extern\s+"[cC]"' \
  internal/monotonic_clock --glob '*.mbt' 2>/dev/null | wc -l | tr -d ' ')
if [ "$extern_count" -ne 1 ]; then
  printf '%s\n' 'monotonic clock ABI must own exactly one foreign declaration' >&2
  failed=1
fi

if ! rg -q '#define LF_MONOTONIC_CLOCK_GETTIME clock_gettime' \
  internal/monotonic_clock/monotonic_clock.c ||
  ! rg -q 'LF_MONOTONIC_CLOCK_GETTIME\s*\(CLOCK_MONOTONIC' \
  internal/monotonic_clock/monotonic_clock.c ||
  ! rg -q 'UINT64_MAX' internal/monotonic_clock/monotonic_clock.c ||
  ! rg -q '^#borrow\(output\)' internal/monotonic_clock/ffi.mbt ||
  ! rg -q 'lunaflux_monotonic_clock_now_millis' \
  internal/monotonic_clock/ffi.mbt \
  internal/monotonic_clock/monotonic_clock.c; then
  printf '%s\n' \
    'monotonic clock ABI lacks exact clock, overflow, or ownership checks' >&2
  failed=1
fi

for binding in \
  'NATIVE_OK : Int = 0|LF_MONOTONIC_CLOCK_OK = 0' \
  'NATIVE_UNAVAILABLE : Int = 1|LF_MONOTONIC_CLOCK_UNAVAILABLE = 1' \
  'NATIVE_OUT_OF_RANGE : Int = 2|LF_MONOTONIC_CLOCK_OUT_OF_RANGE = 2'; do
  moon_pattern=${binding%%|*}
  c_pattern=${binding#*|}
  if ! rg -q "$moon_pattern" internal/monotonic_clock/api.mbt ||
    ! rg -q "$c_pattern" \
      internal/monotonic_clock/monotonic_clock_status.h; then
    printf '%s\n' 'monotonic clock MoonBit/C status vocabulary drifted' >&2
    failed=1
  fi
done

if rg -n 'CLOCK_REALTIME|gettimeofday\s*\(|\btime\s*\(' \
  internal/monotonic_clock runtime/monotonic_clock \
  --glob '*.{c,h,mbt}'; then
  printf '%s\n' 'monotonic clock boundary must never read wall time' >&2
  failed=1
fi

if matches=$(rg -n 'extern\s+"[cC]"' runtime/monotonic_clock \
  --glob '*.mbt' 2>/dev/null); then
  printf '%s\n%s\n' \
    'monotonic clock extern declaration escaped the internal ABI package:' \
    "$matches" >&2
  failed=1
fi

interface_block=$(sed -n \
  '/^pub struct MonotonicClock {/,/^$/p' \
  runtime/monotonic_clock/pkg.generated.mbti)
if [ -z "$interface_block" ] ||
  ! printf '%s\n' "$interface_block" | rg -q '// private fields' ||
  printf '%s\n' "$interface_block" | rg -q 'native|handle|clock_id|origin'; then
  printf '%s\n' 'public monotonic clock capability must remain opaque' >&2
  failed=1
fi

for source_file in \
  internal/monotonic_clock/monotonic_clock.c \
  internal/monotonic_clock/monotonic_clock_probe.c \
  runtime/monotonic_clock/clock.mbt; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s\n' "$source_file" \
      'monotonic clock boundary file exceeds 500 lines' >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux monotonic clock ABI boundary is valid.'
