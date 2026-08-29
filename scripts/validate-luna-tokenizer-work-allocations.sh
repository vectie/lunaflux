#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test \
  tokenizer/luna_work_wbtest.mbt \
  tokenizer/luna_work_reference_wbtest.mbt \
  tokenizer/luna_input_write_wbtest.mbt \
  tokenizer/luna_input_equivalence_wbtest.mbt \
  tokenizer/luna_input_integration_wbtest.mbt \
  tokenizer/sentencepiece_wbtest.mbt \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/tokenizer/tokenizer.whitebox_test.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'Luna tokenizer-work release C output is missing' >&2
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

contains_forbidden_allocation() {
  # Typed exception envelopes are language error plumbing. Production owner,
  # fixed-array, Bytes, Array, and string allocations remain visible.
  rg "$forbidden" |
    rg -v 'moonbit_malloc\(sizeof\(struct _M0DTPC15error5Error' |
    rg -q .
}

# Construction intentionally allocates the owner and all fixed scratch. The
# same extractor and predicate must observe that positive control before the
# warmed progress/reuse surface can prove anything.
positive_body="$(extract_definition 'LunaTokenizerWorker3new(')"
if [ -z "$positive_body" ] ||
  ! printf '%s\n' "$positive_body" | contains_forbidden_allocation; then
  printf '%s\n' 'Luna tokenizer-work allocation positive control is ineffective' >&2
  exit 1
fi

# Every helper reachable from progress is named explicitly. This prevents a
# nonallocating wrapper from hiding allocation in a phase or adjacency scan.
for symbol in \
  'LunaTokenizerWorker18begin__luna__input(' \
  'LunaTokenizerInputWrite15require__worker(' \
  'LunaTokenizerInputWrite10push__byte(' \
  'LunaTokenizerInputWrite6finish(' \
  'LunaTokenizerInputWrite5abort(' \
  'LunaTokenizerWork15require__worker(' \
  'LunaTokenizerWork8progress(' \
  'LunaTokenizerWork12token__count(' \
  'LunaTokenizerWork14was__truncated(' \
  'LunaTokenizerWork16copy__tokens__to(' \
  'LunaTokenizerWork5abort(' \
  'LunaTokenizerWorker14clear__request(' \
  'LunaTokenizerWorker14append__symbol(' \
  'LunaTokenizerWorker21start__special__match(' \
  'LunaTokenizerWorker22finish__special__match(' \
  'LunaTokenizerWorker24progress__special__match(' \
  'LunaTokenizerWorker28sentencepiece__invalid__utf8(' \
  'LunaTokenizerWorker28sentencepiece__scalar__width(' \
  'LunaTokenizerWorker30compare__sentencepiece__scalar(' \
  'LunaTokenizerWorker27find__sentencepiece__scalar(' \
  'LunaTokenizerWorker28start__sentencepiece__scalar(' \
  'sentencepiece__meta__byte(' \
  'LunaTokenizerWorker29finish__sentencepiece__scalar(' \
  'LunaTokenizerWorker31progress__sentencepiece__scalar(' \
  'LunaTokenizerWorker20progress__initialize(' \
  'LunaTokenizerWorker17start__compaction(' \
  'LunaTokenizerWorker27progress__select__adjacency(' \
  'LunaTokenizerWorker22progress__select__rule(' \
  'LunaTokenizerWorker22progress__merge__batch(' \
  'LunaTokenizerWorker18finish__compaction(' \
  'LunaTokenizerWorker20progress__compaction(' \
  'LunaTokenizerWorker13progress__one('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    printf 'Luna tokenizer-work function is missing: %s\n' "$symbol" >&2
    exit 1
  fi
  if printf '%s\n' "$body" | contains_forbidden_allocation; then
    printf 'Luna tokenizer-work warmed path allocates: %s\n' "$symbol" >&2
    exit 1
  fi
done

# The compatibility Bytes facade performs a proportional copy into the fixed
# input backing. It is covered by equivalence, never used as evidence for the
# constant writer path above.
if ! rg -q --pcre2 -U \
  'pub fn LunaTokenizerWorker::begin_bytes(?s).*begin_luna_input(?s).*write\.push_byte(?s).*write\.finish' \
  tokenizer/luna_worker.mbt; then
  printf '%s\n' 'Luna tokenizer Bytes compatibility facade no longer drives the writer' >&2
  exit 1
fi

# Simple scalar getters can be inlined out of release C. If emitted, scan them
# like every other warm function. If absent, require their exact source form;
# `require_worker` is already scanned above, and the remaining operation is a
# scalar field read.
last_work_symbol='LunaTokenizerWork17last__work__units('
last_work_body="$(extract_definition "$last_work_symbol")"
if [ -n "$last_work_body" ]; then
  if printf '%s\n' "$last_work_body" | contains_forbidden_allocation; then
    printf '%s\n' 'Luna tokenizer last-work getter allocates' >&2
    exit 1
  fi
elif ! rg -q --pcre2 -U \
  'pub fn LunaTokenizerWork::last_work_units(?s).*self\.require_worker\(\)\.last_work_units' \
  tokenizer/luna_worker.mbt; then
  printf '%s\n' 'Luna tokenizer last-work getter is neither emitted nor proven scalar' >&2
  exit 1
fi

budget_symbol='LunaTokenizerStepBudget11work__units('
budget_body="$(extract_definition "$budget_symbol")"
if [ -n "$budget_body" ]; then
  if printf '%s\n' "$budget_body" | contains_forbidden_allocation; then
    printf '%s\n' 'Luna tokenizer step-budget getter allocates' >&2
    exit 1
  fi
elif ! rg -q --pcre2 -U \
  'pub fn LunaTokenizerStepBudget::work_units(?s).*self\.work_units' \
  tokenizer/luna_worker_types.mbt; then
  printf '%s\n' 'Luna tokenizer budget getter is neither emitted nor proven scalar' >&2
  exit 1
fi

begin_body="$(extract_definition 'LunaTokenizerWorker18begin__luna__input(')"
if ! printf '%s\n' "$begin_body" | rg -q 'LunaTokenizerInputWrite\)\{'; then
  printf '%s\n' 'Luna tokenizer input begin lost its preallocated valtype writer return' >&2
  exit 1
fi

if ! rg -q 'priv input : FixedArray\[Byte\]' tokenizer/luna_worker_types.mbt ||
  rg -q --pcre2 'priv (?:input|source|request_bytes) : Bytes' \
    tokenizer/luna_worker_types.mbt tokenizer/luna_input_write.mbt ||
  rg -q '^pub fn LunaTokenizer(?:Worker|InputWrite)::(?:input|source|request_bytes|raw_bytes)' \
    tokenizer/pkg.generated.mbti; then
  printf '%s\n' 'Luna tokenizer writer retained or exposed raw Bytes authority' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux bounded tokenizer-work allocation gate passed.'
