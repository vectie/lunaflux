#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon test prefix/radix \
  --target native --release --deny-warn --warn-list +73

generated_c="_build/native/release/test/prefix/radix/radix.whitebox_test.c"
interface="prefix/radix/pkg.generated.mbti"
if [ ! -f "$generated_c" ] || [ ! -f "$interface" ]; then
  printf '%s\n' 'prefix radix release evidence is missing' >&2
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

if ! rg -q 'moonbit_make_bytes\(13, 82\)' "$generated_c"; then
  printf '%s\n' 'prefix radix allocation positive control is ineffective' >&2
  exit 1
fi

hot_body=""
for symbol in \
  'lookup__rooted(' \
  'find__request__root(' \
  'find__child(' \
  'root__index__insert(' \
  'root__index__remove(' \
  'child__index__insert(' \
  'child__index__remove(' \
  'take__free__root(' \
  'take__free__entry(' \
  'take__free__node(' \
  'take__free__token(' \
  'take__free__page(' \
  'publish__pages__have__duplicate(' \
  'protecting__entry(' \
  'refresh__node__protector(' \
  'eviction__candidate(' \
  'eviction__heap__insert(' \
  'eviction__heap__remove(' \
  'normalize__recencies(' \
  'recency__normalization__sift__down(' \
  'compact__changed__path(' \
  'request__root__matches(' \
  'try__plan__publish(' \
  'has__new__root__room(' \
  'existing__publish__preflight(' \
  'mark__expected__adoptions(' \
  'require__publish__plan(' \
  'publish__plan__matches(' \
  'abort__publish(' \
  'insert__rooted(' \
  'attach__pages(' \
  'retain__active(' \
  'release__active(' \
  'require__eviction__plan(' \
  'eviction__plan__matches(' \
  'try__plan__eviction(' \
  'commit__eviction(' \
  'abort__eviction(' \
  'free__planned__label('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then
    source_name="${symbol%\(}"
    source_name="${source_name//__/_}"
    if ! rg -q "^(pub )?fn ([A-Za-z0-9_]+::)?${source_name}\\(" \
        prefix/radix --glob '*.mbt' --glob '!**/*_test.mbt' \
        --glob '!**/*_wbtest.mbt'; then
      printf 'prefix radix allocation function is missing: %s\n' \
        "$symbol" >&2
      exit 1
    fi
    continue
  fi
  hot_body="${hot_body}${body}"
done

# Release compilation is allowed to inline these scalar status and indexed
# buffer accessors. Their exact public surface is pinned below; require their
# source definitions here rather than treating ordinary inlining as absence.
for accessor in \
    'PrefixPublishStart::is_planned' \
    'PrefixPublishStart::is_duplicate' \
    'PrefixPublishStart::is_capacity_exhausted' \
    'PrefixPublishStart::is_rejected' \
    'PrefixPublishStart::duplicate_entry' \
    'PrefixEvictionStart::is_planned' \
    'PrefixEvictionStart::is_absent' \
    'PrefixEvictionStart::is_rejected' \
    'PrefixPublishBuffer::planned_adoption' \
    'PrefixPublishBuffer::was_adopted'; do
  if ! rg -q "^pub fn ${accessor}\\(" prefix/radix --glob '*.mbt'; then
    printf 'prefix radix inlined scalar accessor is missing: %s\n' \
      "$accessor" >&2
    exit 1
  fi
done

if printf '%s\n' "$hot_body" |
  rg -q 'moonbit_make_|moonbit_add_string'; then
  printf '%s\n' 'prefix radix warmed path constructs managed storage' >&2
  exit 1
fi
if printf '%s\n' "$hot_body" |
  rg 'moonbit_malloc' |
  rg -q -v 'PrefixIndexError'; then
  printf '%s\n' 'prefix radix warmed path contains non-error allocation' >&2
  exit 1
fi

for opaque in \
    PrefixIndex \
    PrefixPublishStart \
    PrefixPublishPlan \
    PrefixEvictionStart \
    PrefixEvictionPlan \
    PrefixRequestKey \
    PrefixLookupBuffer \
    PrefixEvictionBuffer \
    PrefixTokenBuffer \
    PrefixPublishBuffer; do
  if ! rg -q --pcre2 -U \
      "^pub struct ${opaque} \\{\\n  // private fields\\n\\}$" \
      "$interface"; then
    printf 'prefix radix authority is not opaque: %s\n' "$opaque" >&2
    exit 1
  fi
done
if [ "$(rg -c ' -> PrefixPublishPlan raise PrefixIndexError$' "$interface")" -ne 1 ] ||
  ! rg -F -x -q \
    'pub fn PrefixIndex::try_plan_publish(Self, PrefixRequestKey, PrefixTokenBuffer, PrefixPublishBuffer, priority~ : Int) -> PrefixPublishStart raise PrefixIndexError' \
    "$interface" ||
  ! rg -F -x -q \
    'pub fn PrefixIndex::publish(Self, PrefixPublishPlan, PrefixTokenBuffer, PrefixPublishBuffer, priority~ : Int) -> PrefixEntryId raise PrefixIndexError' \
    "$interface" ||
  rg -n '^pub fn PrefixPublishPlan::|PrefixPublishPlan.*derive' "$interface"; then
  printf '%s\n' \
    'prefix publication plan return authority or zero-method surface drifted' >&2
  exit 1
fi
publish_start_methods="$(rg '^pub fn PrefixPublishStart::' "$interface" |
  sed -E 's/^pub fn PrefixPublishStart::([^(:]+).*/\1/' | sort)"
if [ "$publish_start_methods" != "$(printf '%s\n' \
    duplicate_entry \
    is_capacity_exhausted \
    is_duplicate \
    is_planned \
    is_rejected \
    plan | sort)" ]; then
  printf '%s\n' 'prefix publication start scalar surface drifted' >&2
  exit 1
fi
for sizing in required_int_cells required_bool_cells required_byte_cells required_reference_cells; do
  if ! rg -F -x -q \
      "pub fn PrefixIndex::${sizing}(PrefixIndexLimits) -> UInt64 raise PrefixIndexError" \
      "$interface"; then
    printf 'prefix radix checked sizing method is missing: %s\n' "$sizing" >&2
    exit 1
  fi
done
if [ "$(rg -c ' -> PrefixEvictionPlan raise PrefixIndexError$' "$interface")" -ne 1 ] ||
  ! rg -F -x -q \
    'pub fn PrefixIndex::try_plan_eviction(Self, PrefixEvictionBuffer) -> PrefixEvictionStart raise PrefixIndexError' \
    "$interface" ||
  ! rg -F -x -q \
    'pub fn PrefixIndex::commit_eviction(Self, PrefixEvictionPlan, PrefixEvictionBuffer) -> PrefixEntryId raise PrefixIndexError' \
    "$interface" ||
  rg -n '^pub fn PrefixEvictionPlan::|PrefixEvictionPlan.*derive' "$interface"; then
  printf '%s\n' \
    'prefix eviction plan return authority or zero-method surface drifted' >&2
  exit 1
fi
if rg -n \
    '^pub struct PrefixIdentity|^pub fn PrefixIdentity::|^pub fn PrefixIndex::(insert|lookup_longest_full_pages|plan_publish|plan_eviction|evict_oldest_zero_reference)\(' \
    "$interface" ||
  rg -n \
    'PublishPlanGenerationExhausted|EvictionPlanGenerationExhausted|NoEvictableEntry' \
    "$interface"; then
  printf '%s\n' 'removed prefix compatibility surface reappeared' >&2
  exit 1
fi
eviction_start_methods="$(rg '^pub fn PrefixEvictionStart::' "$interface" |
  sed -E 's/^pub fn PrefixEvictionStart::([^(:]+).*/\1/' | sort)"
if [ "$eviction_start_methods" != "$(printf '%s\n' \
    is_absent \
    is_planned \
    is_rejected \
    plan | sort)" ]; then
  printf '%s\n' 'prefix eviction start scalar surface drifted' >&2
  exit 1
fi
if rg -n 'impl Debug for (PrefixIndex|PrefixPublishStart|PrefixPublishPlan|PrefixEvictionStart|PrefixEvictionPlan|PrefixRequestKey|PrefixLookupBuffer|PrefixEvictionBuffer|PrefixTokenBuffer|PrefixPublishBuffer)|Map\[' "$interface" ||
  rg -n '@page_allocator\.PageAllocator|retain_cached|release_cached|allocate_run' \
    prefix/radix --glob '*.mbt' --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt' ||
  rg -n 'engine/device|kernels/|device_backend|@device' \
    prefix/radix/moon.pkg prefix/radix/*.mbt; then
  printf '%s\n' 'prefix radix leaked representation or physical page mutation' >&2
  exit 1
fi
if ! rg -F -q 'key.cache_permission() != ReadWrite' \
    prefix/radix/publish_api.mbt ||
  ! rg -F -q 'key.cache_permission() == Disabled' \
    prefix/radix/compressed_lookup.mbt; then
  printf '%s\n' 'prefix radix cache-permission enforcement drifted' >&2
  exit 1
fi
if rg -n \
    'FixedArray::make|Array::|ReadOnlyArray|Prefix(Token|Publish|Lookup|Eviction)Buffer::new' \
    prefix/radix/compressed_insert.mbt \
    prefix/radix/compressed_lookup.mbt \
    prefix/radix/root_index.mbt \
    prefix/radix/child_index.mbt \
    prefix/radix/arena_stacks.mbt \
    prefix/radix/protection.mbt \
    prefix/radix/page_validation.mbt \
    prefix/radix/recency_normalization.mbt \
    prefix/radix/eviction_heap.mbt \
    prefix/radix/publish_api.mbt \
    prefix/radix/eviction.mbt ||
  rg -n '\.as_string\(' \
    prefix/radix/compressed_insert.mbt \
    prefix/radix/compressed_lookup.mbt \
    prefix/radix/publish_api.mbt \
    prefix/radix/eviction.mbt; then
  printf '%s\n' 'prefix radix production transaction core constructs storage' >&2
  exit 1
fi

printf '%s\n' 'LunaFlux compressed prefix radix allocation/source gate passed.'
