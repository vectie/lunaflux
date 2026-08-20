#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test \
  service/framed_wire/luna_request_work_wbtest.mbt \
  service/framed_wire/luna_request_hostile_wbtest.mbt \
  service/framed_wire/luna_request_lifecycle_wbtest.mbt \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/service/framed_wire/framed_wire.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna framed-request release C output is missing' >&2
  exit 1
fi

extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern) > 0 &&
      $0 ~ /^(struct|int|uint|void|double|moonbit_)[A-Za-z0-9_ *]*_M0/ &&
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

forbidden='moonbit_malloc|moonbit_make_|Bytes4make|Array4new|moonbit_add_string'

contains_unexpected_allocation() {
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -v 'moonbit_malloc.*FramedWireError_2e(InvalidFrame|FrameTooLarge|LunaFramedRequestBusy|LunaFramedRequestNotReady|LunaFramedRequestFailed|LunaFramedRequestStale)' |
    rg -q .
}

# Workspace construction intentionally allocates the owner and all fixed
# storage. The same extraction and allocation predicate must see this positive
# control before the warmed scanner surface can prove anything.
positive_body="$(extract_definition 'LunaFramedRequestWorkspace3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_unexpected_allocation; then
  printf '%s\n' 'Luna framed-request allocation positive control is ineffective' >&2
  exit 1
fi

# Every emitted helper reachable from offer/progress is named. Error-constructor
# allocation is language exception plumbing and filtered above; owner, fixed
# array, Bytes, Array, and string allocation remains forbidden.
for symbol in \
  'LunaFramedRequestWorkspace5begin(' \
  'LunaFramedRequestWorkspace16reset__operation(' \
  'request__room(' \
  'request__append(' \
  'read__u32(' \
  'read__i32(' \
  'read__u64(' \
  'LunaFramedRequestWork18require__workspace(' \
  'LunaFramedRequestWork5offer(' \
  'LunaFramedRequestWork5state(' \
  'LunaFramedRequestWork8progress(' \
  'LunaFramedRequestWork7failure(' \
  'LunaFramedRequestWork10take__view(' \
  'LunaFramedRequestWork5abort(' \
  'LunaFramedRequestWorkspace13progress__one(' \
  'LunaFramedRequestWorkspace21progress__prefix__one(' \
  'LunaFramedRequestWorkspace23progress__checksum__one(' \
  'LunaFramedRequestWorkspace21progress__header__one(' \
  'LunaFramedRequestWorkspace24sampling__header__status(' \
  'LunaFramedRequestWorkspace21progress__digest__one(' \
  'LunaFramedRequestWorkspace21progress__layout__one(' \
  'LunaFramedRequestWorkspace20progress__input__one(' \
  'LunaFramedRequestWorkspace26progress__stop__token__one(' \
  'LunaFramedRequestWorkspace37progress__stop__token__duplicate__one(' \
  'LunaFramedRequestWorkspace27progress__stop__string__one(' \
  'LunaFramedRequestWorkspace38progress__stop__string__duplicate__one(' \
  'LunaFramedRequestWorkspace20progress__scope__one(' \
  'LunaFramedRequestWorkspace20progress__trace__one(' \
  'LunaFramedRequestWorkspace11reset__utf8(' \
  'LunaFramedRequestWorkspace13advance__utf8(' \
  'LunaFramedRequestWorkspace10fail__rule(' \
  'LunaFramedRequestWorkspace4fail(' \
  'LunaFramedRequestWorkspace6charge(' \
  'luna__request__safe__identifier__byte(' \
  'luna__request__lower__hex__byte(' \
  'LunaFramedRequestView18require__workspace(' \
  'LunaFramedRequestView7release(' \
  'LunaFramedRequestView15input__byte__at(' \
  'LunaFramedRequestView16input__token__at(' \
  'LunaFramedRequestView25content__digest__byte__at(' \
  'LunaFramedRequestView22plan__digest__byte__at(' \
  'LunaFramedRequestView15stop__token__at(' \
  'LunaFramedRequestView20stop__string__length(' \
  'LunaFramedRequestView22stop__string__byte__at(' \
  'LunaFramedRequestView22cache__scope__byte__at(' \
  'LunaFramedRequestView15trace__byte__at('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna framed-request function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_unexpected_allocation; then
    printf 'Luna framed-request warmed path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

extract_source_definition() {
  local method="$1"
  awk -v pattern="pub fn LunaFramedRequestView::${method}(" '
    index($0, pattern) > 0 { copying = 1; seen = 0 }
    copying {
      print
      opens = gsub(/\{/, "{"); closes = gsub(/\}/, "}")
      depth += opens - closes
      if (opens > 0) seen = 1
      if (seen && depth == 0) exit
    }
  ' service/framed_wire/luna_request_view.mbt
}

# Release may inline simple scalar getters. Scan the emitted body when present;
# otherwise require and scan the exact source definition. This keeps all 32
# public View accessors in the proof without treating optimizer elision as a
# missing function.
while read -r symbol method; do
  body="$(extract_definition "$symbol")"
  if [ -n "$body" ]; then
    if printf '%s\n' "$body" | contains_unexpected_allocation; then
      printf 'Luna framed-request scalar view allocates: %s\n' "$symbol" >&2
      exit 1
    fi
  else
    source_body="$(extract_source_definition "$method")"
    if [ -z "$source_body" ]; then
      printf 'Luna framed-request scalar view is missing: %s\n' "$method" >&2
      exit 1
    fi
    if printf '%s\n' "$source_body" | rg -q \
      'Array|Bytes|String|::make|\.push\(|materialize|@utf8|from_utf8|from_token'; then
      printf 'Luna framed-request inlined scalar view constructs: %s\n' "$method" >&2
      exit 1
    fi
  fi
done <<'EOF'
LunaFramedRequestView6length( length
LunaFramedRequestView17inference__limits( inference_limits
LunaFramedRequestView23protocol__version__wire( protocol_version_wire
LunaFramedRequestView18request__id__value( request_id_value
LunaFramedRequestView11input__kind( input_kind
LunaFramedRequestView13input__length( input_length
LunaFramedRequestView16max__new__tokens( max_new_tokens
LunaFramedRequestView16context__ceiling( context_ceiling
LunaFramedRequestView14sampling__mode( sampling_mode
LunaFramedRequestView21sampling__temperature( sampling_temperature
LunaFramedRequestView10has__top__k( has_top_k
LunaFramedRequestView6top__k( top_k
LunaFramedRequestView10has__top__p( has_top_p
LunaFramedRequestView6top__p( top_p
LunaFramedRequestView21sampling__seed__value( sampling_seed_value
LunaFramedRequestView18stream__preference( stream_preference
LunaFramedRequestView16deadline__millis( deadline_millis
LunaFramedRequestView17cache__permission( cache_permission
LunaFramedRequestView17stop__token__count( stop_token_count
LunaFramedRequestView18stop__string__count( stop_string_count
LunaFramedRequestView19cache__scope__length( cache_scope_length
LunaFramedRequestView13trace__length( trace_length
EOF

# These tiny scalar accessors are commonly inlined out of release C. Their
# callers above are scanned; if no standalone symbol is emitted, require the
# exact scalar-only source form instead of silently dropping the proof.
if ! rg -q --pcre2 -U \
    'pub fn LunaFramedRequestStepBudget::work_units(?s).*self\.work_units' \
    service/framed_wire/luna_request_work_types.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaFramedRequestWork::last_work_units(?s).*require_workspace\(\)\.last_work_units' \
    service/framed_wire/luna_request_work.mbt ||
  ! rg -q --pcre2 -U \
    'pub fn LunaFramedRequestWork::total_work_units(?s).*require_workspace\(\)\.total_work_units' \
    service/framed_wire/luna_request_work.mbt ||
  ! rg -q --pcre2 -U \
    'fn request_invalid\(rule : FramedWireRule\)(?s).*InvalidFrame\(frame=RequestFrame, rule~\)' \
    service/framed_wire/request_codec.mbt; then
  printf '%s\n' 'Luna framed-request inlined scalar helper proof drifted' >&2
  exit 1
fi

# Begin and take_view must retain their fixed-size valtype capability returns.
begin_body="$(extract_definition 'LunaFramedRequestWorkspace5begin(')"
take_body="$(extract_definition 'LunaFramedRequestWork10take__view(')"
if ! printf '%s\n' "$begin_body" | rg -q 'LunaFramedRequestWork' ||
  ! printf '%s\n' "$take_body" | rg -q 'LunaFramedRequestView'; then
  printf '%s\n' 'Luna framed-request capability return lost valtype evidence' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux bounded framed-request allocation gate passed.'
