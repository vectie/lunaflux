#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test service/request_admission \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/request_admission/request_admission.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'request semantic-migration release C output is missing' >&2
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

allocation_lines() {
  rg 'moonbit_malloc|moonbit_make_|Bytes4make|moonbit_add_string' || true
}

storage_new="$(extract_definition 'LunaRequestSemanticStorage3new(')"
if [ -z "$storage_new" ] ||
  ! printf '%s\n' "$storage_new" | allocation_lines | rg -q .; then
  printf '%s\n' 'semantic-migration allocation positive control is ineffective' >&2
  exit 1
fi

# These helpers are nonraising scalar/read-only operations in the inner
# cooperative closure. No error-constructor or Failure-name filter is applied.
for symbol in \
  'luna__semantic__utf8__width(' \
  'luna__semantic__utf8__byte(' \
  'luna__semantic__codepoint__at(' \
  'luna__semantic__codepoint__units__at(' \
  'progress__semantic__cache__measure(' \
  'progress__semantic__string__measure(' \
  'begin__semantic__import(' \
  'LunaPreparedRequestClaim15is__stop__token(' \
  'LunaRequestStopTokenView15is__stop__token(' \
  'LunaRequestSemanticStorage19stop__token__status('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'semantic-migration scalar function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | allocation_lines | rg -q .; then
    printf 'semantic-migration scalar function allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

for phase in \
  progress_semantic_cache_measure \
  progress_semantic_begin \
  progress_semantic_token_copy \
  progress_semantic_string_measure \
  progress_semantic_string_header \
  progress_semantic_string_emit \
  progress_semantic_cache_emit \
  progress_semantic_finish \
  progress_semantic_validate \
  progress_semantic_take; do
  if ! rg -q "fn LunaRequestPreparationPool::${phase}" \
      service/request_admission/pool_semantic_progress.mbt ||
    ! rg -q "self\.${phase}\(lane\)" \
      service/request_admission/pool_progress.mbt; then
    printf 'semantic-migration cooperative phase is not owned by progress: %s\n' \
      "$phase" >&2
    exit 1
  fi
done

if rg -n \
    '@utf8\.encode|token_only\(|IncrementalOutput::new\(|output_workspace\.begin\(' \
    service/request_admission/admit.mbt \
    service/request_admission/pool.mbt \
    service/request_admission/pool_progress.mbt \
    service/request_admission/pool_semantic_progress.mbt \
    service/request_admission/semantic_import.mbt; then
  printf '%s\n' \
    'request semantic migration restored a proportional/raw legacy path' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'progress_semantic_string_measure(?s:.*)luna_semantic_codepoint_at(?s:.*)semantic_unit_offset \+= luna_semantic_codepoint_units_at(?s:.*)semantic_byte_length \+= luna_semantic_utf8_width' \
    service/request_admission/pool_semantic_progress.mbt ||
  ! rg -q --pcre2 -U \
    'progress_semantic_string_emit(?s:.*)push_stop_string_byte(?s:.*)semantic_codepoint_byte_index \+= 1' \
    service/request_admission/pool_semantic_progress.mbt ||
  ! rg -q --pcre2 -U \
    'progress_semantic_validate(?s:.*)require_semantic_work\(\)\.progress\(\)' \
    service/request_admission/pool_semantic_progress.mbt ||
  ! rg -q --pcre2 -U \
    'progress_semantic_take(?s:.*)take_lease\(\)(?s:.*)begin_luna_request_semantics' \
    service/request_admission/pool_semantic_progress.mbt; then
  printf '%s\n' 'request semantic migration no longer performs one bounded step' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'fn LunaPreparedRequestClaim::release_resources_only(?s:.*?)self\.semantic_lease\[0\]\.release\(\) catch(?s:.*?)self\.semantic_lease\.clear\(\)(?s:.*?)self\.scheduler_request = None(?s:.*?)output\.release\(\)(?s:.*?)lease\.release\(\)' \
    service/request_admission/types.mbt; then
  printf '%s\n' \
    'claim cleanup can mutate lower resources before semantic release proof' >&2
  exit 1
fi

if ! rg -q --pcre2 -U \
    'semantic_int_cells(?s:.*)checked_add_cells(?s:.*)semantic_byte_cells(?s:.*)checked_add_cells(?s:.*)checked_multiply_cells\(per_lane_int, lanes\)(?s:.*)checked_multiply_cells\(per_lane_byte, lanes\)' \
    service/request_admission/pool_storage.mbt; then
  printf '%s\n' 'semantic storage is outside checked aggregate lane accounting' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux request semantic-migration allocation gate passed.'
