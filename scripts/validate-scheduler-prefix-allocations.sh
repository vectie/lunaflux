#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test scheduler/core \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/scheduler/core/core.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'scheduler prefix release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
      $0 ~ /\($/ {
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
  ' "$generated_c"
}

if ! rg -q 'moonbit_make_bytes\(23, 80\)' "$generated_c"; then
  printf '%s\n' 'scheduler prefix allocation positive control is ineffective' >&2
  exit 1
fi

hot_body=""
for symbol in \
  'discover__prefix__candidate(' \
  'copy__prefix__tokens(' \
  'prefix__hit__preflight(' \
  'revalidate__waiting__prefix(' \
  'acquire__revalidated__prefix(' \
  'rollback__selected__prefixes(' \
  'commit__selected__prefixes(' \
  'stage__selected__prefix__pages(' \
  'build__next(' \
  'release__prefix__entry(' \
  'release__waiting__prefix(' \
  'preflight__table__page__release(' \
  'preflight__prefix__entry__release(' \
  'preflight__waiting__prefix__device__loss(' \
  'release__waiting__prefix__device__state(' \
  'clear__cached__prefix__device__state(' \
  'evict__one__prefix(' \
  'evict__for__virtual__room(' \
  'prefix__publish__preflight(' \
  'commit__prompt__prefix__plan(' \
  'publish__prompt__prefix(' \
  'waiting__successor(' \
  'waiting__selection__precedes(' \
  'waiting__selection__sift__down(' \
  'prepare__waiting__selection(' \
  'pop__waiting__selection(' \
  'active__cursor__slot(' \
  'active__cursor__position(' \
  'enqueue__waiting(' \
  'remove__waiting(' \
  'add__active(' \
  'remove__active(' \
  'deadline__precedes(' \
  'deadline__swap(' \
  'deadline__sift__up(' \
  'deadline__sift__down(' \
  'deadline__remove(' \
  'record__prompt__compute(' \
  'prefill__start(' \
  'has__prefix__key(' \
  'prefix__key('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    source_name="${symbol%\(}"
    source_name="${source_name//__/_}"
    if ! rg -q "^(pub )?fn ([A-Za-z0-9_]+::)?${source_name}\\(" \
        scheduler/core --glob '*.mbt' --glob '!**/*_test.mbt' \
        --glob '!**/*_wbtest.mbt'; then
      printf 'scheduler prefix allocation function is missing: %s\n' \
        "$symbol" >&2
      exit 1
    fi
    continue
  fi
  hot_body="${hot_body}${body}"
done

if [ "$(rg -c '(self|scheduler)\.rollback_selected_prefixes\(\)' \
    scheduler/core/planning.mbt)" -lt 5 ] ||
  ! rg -q -U \
    'block_tables\.checkpoint\(\) catch \{[[:space:]]*error => \{[[:space:]]*self\.rollback_selected_prefixes\(\)' \
    scheduler/core/planning.mbt ||
  ! rg -q -U \
    'page_allocator\.checkpoint\(\) catch \{[[:space:]]*error => \{[[:space:]]*self\.rollback_selected_prefixes\(\)' \
    scheduler/core/planning.mbt ||
  ! rg -q -U \
    'plan\.checkpoint\(\) catch \{[[:space:]]*error => \{[[:space:]]*self\.rollback_selected_prefixes\(\)' \
    scheduler/core/planning.mbt; then
  printf '%s\n' \
    'scheduler checkpoint acquisition no longer releases selected prefix refs' >&2
  exit 1
fi

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'scheduler prefix path constructs managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'SchedulerError|PrefixIndexError|InferenceContractError'; then
  printf '%s\n' 'scheduler prefix path contains non-error allocation' >&2
  exit 1
fi

if rg -n \
    'FixedArray\[@radix\.PrefixEntryId\?\]|PrefixEntryId\?|LunaRequestSemanticView|StopConditions|CachePolicy|stop_string|PrefixIdentity|lookup_longest_full_pages|evict_oldest_zero_reference' \
    scheduler/core --glob '*.mbt' --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt' ||
  ! rg -q 'priv prefix_entry_indices : FixedArray\[Int\]' \
    scheduler/core/owner_types.mbt ||
  ! rg -q 'priv prefill_cursors : FixedArray\[Int\]' \
    scheduler/core/owner_types.mbt; then
  printf '%s\n' 'scheduler prefix authority or scalar storage drifted' >&2
  exit 1
fi
if rg -F -n -e '.plan_publish(' -e '.plan_eviction(' \
    scheduler/core/prefix_ownership.mbt \
    scheduler/core/prefix_publication.mbt \
    scheduler/core/prefix_device_loss.mbt \
    scheduler/core/planning_selection.mbt \
    scheduler/core/waiting_selection_heap.mbt \
    scheduler/core/waiting_registry.mbt \
    scheduler/core/active_registry.mbt \
    scheduler/core/deadline_heap.mbt \
    scheduler/core/preemption.mbt ||
  ! rg -F -q '.try_plan_publish(' scheduler/core/prefix_publication.mbt ||
  ! rg -F -q '.try_plan_eviction(' scheduler/core/prefix_publication.mbt; then
  printf '%s\n' \
    'scheduler warmed prefix pressure must use scalar start statuses' >&2
  exit 1
fi
if rg -n \
    'FixedArray::make|Array::|ReadOnlyArray|Bytes::|String::|PrefixIdentity|lookup_longest_full_pages|evict_oldest_zero_reference' \
    scheduler/core/prefix_ownership.mbt \
    scheduler/core/prefix_publication.mbt \
    scheduler/core/prefix_device_loss.mbt \
    scheduler/core/planning_selection.mbt \
    scheduler/core/waiting_selection_heap.mbt \
    scheduler/core/waiting_registry.mbt \
    scheduler/core/active_registry.mbt \
    scheduler/core/deadline_heap.mbt \
    scheduler/core/preemption.mbt; then
  printf '%s\n' 'scheduler prefix/recovery core constructs storage or uses legacy APIs' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux scheduler prefix allocation/source gate passed.'
