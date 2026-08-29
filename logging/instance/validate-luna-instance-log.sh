#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

moon test logging/instance \
  --target native --release --deny-warn --warn-list +73

blackbox_c="_build/native/release/test/logging/instance/instance.blackbox_test.c"
whitebox_c="_build/native/release/test/logging/instance/instance.whitebox_test.c"
interface="logging/instance/pkg.generated.mbti"
if [ ! -f "$blackbox_c" ] || [ ! -f "$whitebox_c" ] ||
  [ ! -f "$interface" ]; then
  printf '%s\n' 'instance log release evidence is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  local source="$2"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      ($0 ~ /^struct moonbit_result_/ || $0 ~ /^struct _M0TP/ ||
       $0 ~ /^int32_t / || $0 ~ /^uint32_t / ||
       $0 ~ /^uint64_t / || $0 ~ /^void /) {
      candidate = 1; body = $0 ORS; next
    }
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
  ' "$source"
}

constructor="$(extract_definition 'LunaInstanceLog3new(' "$blackbox_c")"
if [ -z "$constructor" ]; then
  printf '%s\n' 'instance log constructor is missing' >&2
  exit 1
fi
if [ "$(printf '%s\n' "$constructor" |
    grep -Fc 'moonbit_make_int32_array(')" -ne 8 ] ||
  [ "$(printf '%s\n' "$constructor" |
    grep -Fc 'moonbit_make_int64_array(')" -ne 2 ] ||
  [ "$(printf '%s\n' "$constructor" |
    grep -Fc 'LunaInstanceLog*)moonbit_malloc(sizeof(struct')" -ne 1 ]; then
  printf '%s\n' 'instance log startup storage formula drifted' >&2
  exit 1
fi

hot_body=""
for symbol in \
  'LunaInstanceLog6record(' \
  'LunaInstanceLog9event__at(' \
  'LunaInstanceLog10reason__at(' \
  'LunaInstanceLog21timestamp__millis__at(' \
  'LunaInstanceLog9count__at(' \
  'LunaInstanceLog20duration__millis__at(' \
  'LunaInstanceLog14logical__index(' \
  'LunaInstanceLog8snapshot(' \
  'LunaInstanceLogSnapshot9event__at(' \
  'LunaInstanceLogSnapshot6length(' \
  'LunaInstanceLogSnapshot13require__live(' \
  'LunaInstanceLogSnapshot14require__index(' \
  'luna__instance__log__event__code(' \
  'luna__instance__log__reason__code(' \
  'luna__instance__log__event__from__code(' \
  'luna__instance__log__reason__from__code('; do
  body="$(extract_definition "$symbol" "$blackbox_c")"
  if [ -z "$body" ]; then
    printf 'instance log allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string|memcpy|memmove'; then
  printf '%s\n' 'instance log warmed path constructs or bulk-copies storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'LunaInstanceLog(Error|StaleSnapshot|SnapshotGenerationExhausted)'; then
  printf '%s\n' 'instance log warmed path contains non-error allocation' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes\(19, 77\)' "$whitebox_c"; then
  printf '%s\n' 'instance log allocation positive control is ineffective' >&2
  exit 1
fi

enum_variants() {
  local type="$1"
  awk -v type="$type" '
    $0 == "pub(all) enum " type " {" { inside = 1; next }
    inside && $0 ~ /^} derive\(/ { exit }
    inside { print }
  ' "$interface"
}

expected_events=$'  LunaInstanceStarting\n  LunaInstanceReady\n  LunaAdmissionAccepted\n  LunaAdmissionRejected\n  LunaRequestCompleted\n  LunaRequestCancelled\n  LunaRequestDeadline\n  LunaRequestFailed\n  LunaWorkerRestarted\n  LunaWorkerFailed\n  LunaNetworkAccepted\n  LunaNetworkDisconnected\n  LunaNetworkRejected\n  LunaBackpressureObserved\n  LunaDrainStarted\n  LunaInstanceClosed'
expected_reasons=$'  LunaLogNoReason\n  LunaLogAuthentication\n  LunaLogValidation\n  LunaLogCapacity\n  LunaLogDeadline\n  LunaLogCancellation\n  LunaLogWorkerExit\n  LunaLogWorkerProtocol\n  LunaLogNetworkRead\n  LunaLogNetworkWrite\n  LunaLogPeerClosed\n  LunaLogServiceDraining\n  LunaLogInternal'
expected_rules=$'  LunaInstanceLogCapacityRule\n  LunaInstanceLogIndexRule\n  LunaInstanceLogTimestampRule\n  LunaInstanceLogCountRule\n  LunaInstanceLogDurationRule\n  LunaInstanceLogSnapshotRule'
if [ "$(enum_variants LunaInstanceLogEvent)" != "$expected_events" ] ||
  [ "$(enum_variants LunaInstanceLogReason)" != "$expected_reasons" ] ||
  [ "$(enum_variants LunaInstanceLogRule)" != "$expected_rules" ]; then
  printf '%s\n' 'instance log finite vocabulary drifted' >&2
  exit 1
fi

for opaque_type in LunaInstanceLog LunaInstanceLogSnapshot; do
  if [ "$(grep -Fxc "pub struct ${opaque_type} {" "$interface")" -ne 1 ] ||
    ! awk -v type="$opaque_type" '
      $0 == "pub struct " type " {" { seen = 1; next }
      seen && $0 == "  // private fields" { private_fields = 1; next }
      seen && $0 == "}" { exit !(private_fields == 1) }
    ' "$interface"; then
    printf 'instance log type is not private-field opaque: %s\n' \
      "$opaque_type" >&2
    exit 1
  fi
done

if rg -q 'impl Debug for LunaInstanceLog|String|Bytes|FixedArray|Array\[|Map\[' \
    "$interface" ||
  rg -n 'String|Bytes|Map\[|RequestId|ModelPath|Pointer|Debug' \
    logging/instance/types.mbt logging/instance/vocabulary.mbt \
    logging/instance/owner.mbt logging/instance/snapshot.mbt; then
  printf '%s\n' 'instance log leaks payload, identity, raw storage, or Debug' >&2
  exit 1
fi

if [ "$(grep -c '^pub fn LunaInstanceLog::' "$interface")" -ne 14 ] ||
  [ "$(grep -c '^pub fn LunaInstanceLogSnapshot::' "$interface")" -ne 8 ] ||
  [ "$(grep -c '^pub fn ' "$interface")" -ne 22 ]; then
  printf '%s\n' 'instance log public method surface drifted' >&2
  exit 1
fi

for signature in \
  'pub fn LunaInstanceLog::record(Self, LunaInstanceLogEvent, LunaInstanceLogReason, timestamp_millis~ : UInt64, count~ : Int, duration_millis~ : Int) -> Unit raise LunaInstanceLogError' \
  'pub fn LunaInstanceLog::snapshot(Self) -> LunaInstanceLogSnapshot raise LunaInstanceLogError' \
  'pub fn LunaInstanceLogSnapshot::event_at(Self, Int) -> LunaInstanceLogEvent raise LunaInstanceLogError' \
  'pub fn LunaInstanceLogSnapshot::timestamp_millis_at(Self, Int) -> UInt64 raise LunaInstanceLogError'; do
  if [ "$(grep -Fxc "$signature" "$interface")" -ne 1 ]; then
    printf 'instance log critical signature drifted: %s\n' "$signature" >&2
    exit 1
  fi
done

printf '%s\n' \
  'LunaFlux bounded instance log allocation/source gate passed.'
