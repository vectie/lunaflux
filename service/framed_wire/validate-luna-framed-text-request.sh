#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
moon test service/framed_wire --target native --release --deny-warn --warn-list +73
cfile="_build/native/release/test/service/framed_wire/framed_wire.whitebox_test.c"
extract_definition() {
  local pattern="$1"
  awk -v pattern="$pattern" '
    index($0, pattern)>0 && $0~/_M0/ && $0~/\($/ { c=1; b=$0 ORS; next }
    c { b=b $0 ORS; if($0~/^\);$/){c=0;b="";next} if($0~/^\) \{$/){p=1;d=1;printf "%s",b;c=0;next} }
    p { print; o=gsub(/\{/,"{"); x=gsub(/\}/,"}"); d+=o-x; if(d==0)exit }
  ' "$cfile"
}
for symbol in \
  'LunaFramedTextRequestWorkspace11begin__text(' \
  'LunaFramedTextRequestWorkspace36begin__text__with__sampling__scalars(' \
  'LunaFramedTextRequestWrite10push__byte(' \
  'LunaFramedTextRequestWrite17push__stop__token(' \
  'LunaFramedTextRequestWrite26push__stop__string__length(' \
  'LunaFramedTextRequestWrite24push__stop__string__byte(' \
  'LunaFramedTextRequestWrite6finish(' \
  'LunaFramedTextRequestWork8progress(' \
  'LunaFramedTextRequestWorkspace13progress__one(' \
  'LunaFramedTextRequestWorkspace12header__byte(' \
  'LunaFramedTextRequestWork10take__view(' \
  'LunaFramedTextRequestView15copy__chunk__to(' \
  'LunaFramedTextRequestView7release('; do
  body="$(extract_definition "$symbol")"
  if [ -z "$body" ]; then printf 'missing framed text symbol: %s\n' "$symbol" >&2; exit 1; fi
  if printf '%s\n' "$body" | rg 'moonbit_malloc|moonbit_make_|moonbit_add_string|memcpy|memmove' |
    rg -v 'moonbit_malloc.*FramedWireError' | rg -q .; then
    printf 'framed text warmed path allocates/copies proportionally: %s\n' "$symbol" >&2; exit 1
  fi
done
positive="$(extract_definition 'LunaFramedTextRequestWorkspace3new(')"
if [ -z "$positive" ] || ! printf '%s\n' "$positive" | rg -q 'moonbit_malloc|moonbit_make_'; then
  printf '%s\n' 'framed text allocation positive control failed' >&2; exit 1
fi
mbti="service/framed_wire/pkg.generated.mbti"
for type in LunaFramedTextRequestStepBudget LunaFramedTextRequestWorkspace LunaFramedTextRequestWrite LunaFramedTextRequestWork LunaFramedTextRequestView; do
  rg -U -q "pub struct ${type} \{\n  // private fields\n\}" "$mbti" || { printf 'nonopaque framed text type: %s\n' "$type" >&2; exit 1; }
  ! rg -q "impl Debug for ${type}" "$mbti" || { printf 'Debug framed text authority: %s\n' "$type" >&2; exit 1; }
done
if rg -q -- 'LunaFramedTextRequest.*-> (FixedArray|Array|Bytes|String|@inference\.GenerateRequest)' "$mbti"; then
  printf '%s\n' 'framed text raw/object authority leaked' >&2; exit 1
fi
if [ "$(rg -c '^pub fn LunaFramedTextRequestWorkspace::' "$mbti")" -ne 5 ] ||
  [ "$(rg -c '^pub fn LunaFramedTextRequestWrite::' "$mbti")" -ne 6 ] ||
  [ "$(rg -c '^pub fn LunaFramedTextRequestWork::' "$mbti")" -ne 7 ] ||
  [ "$(rg -c '^pub fn LunaFramedTextRequestView::' "$mbti")" -ne 3 ]; then
  printf '%s\n' 'framed text method surface drifted' >&2; exit 1
fi
printf '%s\n' 'Luna framed text request allocation and API gate passed.'
