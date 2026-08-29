#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test metrics/instance \
  --target native --release --deny-warn --warn-list +73

blackbox_c="_build/native/release/test/metrics/instance/instance.blackbox_test.c"
whitebox_c="_build/native/release/test/metrics/instance/instance.whitebox_test.c"
interface="metrics/instance/pkg.generated.mbti"
if [ ! -f "$blackbox_c" ] || [ ! -f "$whitebox_c" ] ||
  [ ! -f "$interface" ]; then
  printf '%s\n' 'instance metrics release evidence is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  local source="$2"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      ($0 ~ /^struct moonbit_result_/ || $0 ~ /^struct _M0TP/ ||
       $0 ~ /^int32_t / || $0 ~ /^uint64_t /) {
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

constructor="$(extract_definition 'LunaInstanceMetrics3new(' "$blackbox_c")"
if [ -z "$constructor" ]; then
  printf '%s\n' 'instance metrics constructor is missing' >&2
  exit 1
fi
if [ "$(printf '%s\n' "$constructor" | grep -Fc 'moonbit_make_int64_array(22, 0ull)')" -ne 2 ] ||
  [ "$(printf '%s\n' "$constructor" | grep -Fc 'moonbit_make_int32_array(6, 0)')" -ne 2 ] ||
  [ "$(printf '%s\n' "$constructor" | grep -Fc 'moonbit_make_int64_array(96, 0ull)')" -ne 2 ] ||
  [ "$(printf '%s\n' "$constructor" | grep -Fc 'moonbit_malloc(sizeof(struct')" -ne 1 ]; then
  printf '%s\n' 'instance metrics startup storage formula drifted' >&2
  exit 1
fi

hot_body=""
for symbol in \
  'record__counter_2einner(' \
  'record__counter__u64(' \
  'set__gauge(' \
  'record__histogram(' \
  'counter__value(' \
  'gauge__value(' \
  'histogram__bucket__value(' \
  'LunaInstanceMetrics5reset(' \
  'LunaInstanceMetrics8snapshot(' \
  'luna__instance__histogram__bucket__upper__bound(' \
  'LunaInstanceMetricsSnapshot14counter__value(' \
  'LunaInstanceMetricsSnapshot12gauge__value(' \
  'LunaInstanceMetricsSnapshot24histogram__bucket__value(' \
  'LunaInstanceMetricsSnapshot13require__live(' \
  'saturating__add(' \
  'histogram__bucket__index(' \
  'histogram__bucket__upper__bound__unchecked(' \
  'histogram__cell__index('; do
  body="$(extract_definition "$symbol" "$blackbox_c")"
  if [ -z "$body" ]; then
    printf 'instance metrics allocation function is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
  hot_body="${hot_body}${body}"
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'instance metrics warmed path constructs managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'LunaInstanceMetricsError'; then
  printf '%s\n' 'instance metrics warmed path contains non-error allocation' >&2
  exit 1
fi
if ! rg -q 'moonbit_make_bytes\(19, 77\)' "$whitebox_c"; then
  printf '%s\n' 'instance metrics allocation positive control is ineffective' >&2
  exit 1
fi

expected_counters=$'  Admissions\n  Completions\n  Cancellations\n  Deadlines\n  Failures\n  PromptTokens\n  GeneratedTokens\n  WorkerRestarts\n  WorkerFailures\n  NetworkAccepts\n  NetworkDisconnects\n  NetworkRejections\n  Backpressure\n  PrefixLookups\n  PrefixHits\n  PrefixMisses\n  PrefixEvictions\n  PrefixTokensReused\n  PrefixTokensComputed\n  PrefixPublications\n  GraphHits\n  GraphMisses'
expected_gauges=$'  QueueDepth\n  ActiveRequests\n  KvPagesUsed\n  KvPagesFree\n  PrefixEntries\n  PrefixPages'
expected_histograms=$'  RequestLatencyMillis\n  FirstTokenLatencyMillis\n  InterTokenLatencyMillis\n  BatchRows\n  BatchTokens'
expected_histograms=$'  RequestLatencyMillis\n  FirstTokenLatencyMillis\n  InterTokenLatencyMillis\n  ColdStartLatencyMillis\n  BatchRows\n  BatchTokens'

enum_variants() {
  local type="$1"
  awk -v type="$type" '
    $0 == "pub(all) enum " type " {" { inside = 1; next }
    inside && $0 == "}" { exit }
    inside { print }
  ' "$interface"
}

if [ "$(enum_variants LunaInstanceCounter)" != "$expected_counters" ] ||
  [ "$(enum_variants LunaInstanceGauge)" != "$expected_gauges" ] ||
  [ "$(enum_variants LunaInstanceHistogram)" != "$expected_histograms" ]; then
  printf '%s\n' 'instance metric vocabulary drifted' >&2
  exit 1
fi

for opaque_type in LunaInstanceMetrics LunaInstanceMetricsSnapshot; do
  if [ "$(grep -Fxc "pub struct ${opaque_type} {" "$interface")" -ne 1 ] ||
    ! awk -v type="$opaque_type" '
      $0 == "pub struct " type " {" { seen = 1; next }
      seen && $0 == "  // private fields" { private_fields = 1; next }
      seen && $0 == "}" { exit !(private_fields == 1) }
    ' "$interface"; then
    printf 'instance metrics type is not private-field opaque: %s\n' \
      "$opaque_type" >&2
    exit 1
  fi
done

if rg -q 'impl Debug for LunaInstance|String|Bytes|FixedArray|Array\[|Map\[' \
    "$interface" ||
  rg -n 'String|Bytes|Map\[|label' \
    metrics/instance/types.mbt metrics/instance/vocabulary.mbt \
    metrics/instance/owner.mbt metrics/instance/snapshot.mbt; then
  printf '%s\n' 'instance metrics leaks payload, labels, raw storage, or Debug' >&2
  exit 1
fi

if [ "$(grep -c '^pub fn LunaInstanceMetrics::' "$interface")" -ne 10 ] ||
  [ "$(grep -c '^pub fn LunaInstanceMetricsSnapshot::' "$interface")" -ne 4 ] ||
  [ "$(grep -c '^pub fn ' "$interface")" -ne 19 ]; then
  printf '%s\n' 'instance metrics public method surface drifted' >&2
  exit 1
fi

printf '%s\n' \
  'LunaFlux bounded instance metrics allocation/source gate passed.'
