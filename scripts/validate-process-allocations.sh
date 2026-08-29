#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test internal/process \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/internal/process/process.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'process release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    $0 ~ pattern && $0 ~ /^struct moonbit_result_/ {
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
  '12ChildProcess.*26begin__read__frame_2einner' \
  '12ChildProcess.*21progress__read__frame' \
  '12ChildProcess.*27begin__write__frame_2einner' \
  '12ChildProcess.*22progress__write__frame' \
  '12ChildProcess.*19read__exact_2einner' \
  '12ChildProcess.*20write__exact_2einner' \
  '12ChildProcess.*26read__frame__with__timeout' \
  '12ChildProcess.*27write__frame__with__timeout' \
  '16InheritedChannel.*begin__read__frame__internal' \
  '16InheritedChannel.*progress__read__frame__internal' \
  '16InheritedChannel.*begin__write__frame_2einner' \
  '16InheritedChannel.*progress__write__frame' \
  '16InheritedChannel.*11read__exact' \
  '16InheritedChannel.*12write__exact' \
  '16InheritedChannel.*20read__frame__or__eof' \
  '16InheritedChannel.*12write__frame'; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'process allocation gate function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

# Typed ProcessError values allocate only on exceptional branches. Reject
# every collection/string/ref construction and every other heap allocation in
# the complete generated cooperative parent and blocking child frame paths.
# The latter is the production serialized device-child request loop; captured
# defer environments there would otherwise allocate on every successful frame.
if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_.*array|moonbit_make_bytes|moonbit_make_ref|moonbit_add_string'; then
  printf '%s\n' 'process pending-frame path constructs a collection, ref, or string' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'ProcessError'; then
  printf '%s\n' 'process pending-frame path contains a non-error heap allocation' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" | rg -q '_2adefer'; then
  printf '%s\n' \
    'process frame path contains optimizer-dependent deferred cleanup' >&2
  exit 1
fi
if rg -q 'PendingFrame(Read|Write)' "$generated_c"; then
  printf '%s\n' 'process pending-frame path still materializes a frame token' >&2
  exit 1
fi

# Positive control: a dedicated white-box helper constructs a 13-byte fixed
# array. If this disappears, the release-C scan is no longer observing the
# allocation vocabulary it is intended to reject.
if ! rg -q 'moonbit_make_bytes\(13, 90\)' "$generated_c"; then
  printf '%s\n' 'process allocation positive control is ineffective' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux process pending-frame allocation gate passed.'
