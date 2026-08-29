#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test internal/process \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/internal/process/process.whitebox_test.c"
interface="internal/process/pkg.generated.mbti"
if [ ! -f "$generated_c" ] || [ ! -f "$interface" ]; then
  printf '%s\n' 'process release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 && $0 ~ /^struct moonbit_result_|^int32_t |^int64_t |^struct _M0TP/ {
      candidate = 1
      body = $0 ORS
      next
    }
    candidate {
      body = body $0 ORS
      if ($0 ~ /^\);$/) { candidate = 0; body = ""; next }
      if ($0 ~ /^\) \{$/) {
        copying = 1
        depth = 1
        printf "%s", body
        candidate = 0
        next
      }
    }
    copying {
      print
      opens = gsub(/\{/, "{")
      closes = gsub(/\}/, "}")
      depth += opens - closes
      if (depth == 0) exit
    }
  ' "$generated_c"
}

hot_body=""
for symbol in \
  '19observe__idle__exit(' \
  '26begin__write__frame__until(' \
  '25begin__read__frame__until(' \
  '33begin__write__frame__at__deadline(' \
  '32begin__read__frame__at__deadline(' \
  '22admit__frame__deadline(' \
  '28begin__shutdown__maintenance(' \
  '31progress__shutdown__maintenance(' \
  '27progress__maintenance__poll(' \
  '26maintenance__clock__status(' \
  '30maintenance__clock__status__at(' \
  '25record__maintenance__exit(' \
  '21maintenance__deadline(' \
  '26apply__final__reap__result(' \
  '28begin__final__reap__interval(' \
  '29expire__final__reap__interval(' \
  '31progress__final__reap__interval(' \
  '21progress__stuck__reap(' \
  '17capture__deadline(' \
  '25clear__pending__transfers(' \
  '21clear__pending__write(' \
  '20clear__pending__read(' \
  '30maintenance__remaining__millis(' \
  '17maintenance__exit('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'process maintenance allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string|moonbit_array_copy|memcpy|memmove'; then
  printf '%s\n' \
    'cooperative process maintenance constructs or copies managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'ProcessError'; then
  printf '%s\n' \
    'cooperative process maintenance contains a non-error heap allocation' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" | rg -q 'moonbit_make_ref'; then
  printf '%s\n' 'cooperative process polling still allocates an out-Ref' >&2
  exit 1
fi

if ! rg -q 'moonbit_make_bytes\(13, 90\)' "$generated_c"; then
  printf '%s\n' 'process maintenance allocation positive control is ineffective' >&2
  exit 1
fi

if ! rg -q 'int64_t lunaflux_process_maintenance_try_wait' \
    internal/process/process_maintenance.c ||
  rg -q 'int32_t \*kind|int32_t \*code' \
    internal/process/process_maintenance.c; then
  printf '%s\n' 'maintenance wait evidence must remain a packed scalar' >&2
  exit 1
fi

if ! grep -Fqx \
    'pub fn ChildProcessMaintenanceProgress::is_cleanup_stuck(Self) -> Bool' \
    "$interface"; then
  printf '%s\n' 'process retained cleanup-stuck API drifted' >&2
  exit 1
fi

if ! grep -Fq 'pub fn ChildProcess::begin_write_frame_until' "$interface" ||
  ! grep -Fq 'pub fn ChildProcess::begin_read_frame_until' "$interface"; then
  printf '%s\n' 'process absolute frame-deadline API drifted' >&2
  exit 1
fi

if ! grep -Fqx \
    'pub fn ChildProcess::observe_idle_exit(Self) -> ChildProcessIdleObservation raise ProcessError' \
    "$interface" ||
  ! grep -Fqx \
    'pub fn ChildProcessIdleObservation::exit(Self) -> ChildExit raise ProcessError' \
    "$interface"; then
  printf '%s\n' 'process idle-exit observation API drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux cooperative process maintenance allocation gate passed.'
