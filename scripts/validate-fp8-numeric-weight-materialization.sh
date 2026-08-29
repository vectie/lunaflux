#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

header=model/numeric_weight_file/header.mbt
payload=model/numeric_weight_file/validate_payload.mbt
errors=model/numeric_weight_file/errors.mbt
test_file=model/numeric_weight_file/fp8_weight_file_wbtest.mbt
package=model/numeric_weight_file/moon.pkg

for file in "$header" "$payload" "$errors" "$test_file" "$package"; do
  [ -f "$file" ] || {
    printf '%s\n' "missing FP8 numeric materialization source: $file" >&2
    exit 1
  }
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || {
    printf '%s\n' "FP8 numeric materialization file exceeds budget: $file ($lines)" >&2
    exit 1
  }
done

for anchor in \
  '@numeric_contract.StorageDType::f8_e4m3_finite()' \
  '"F8_E4M3"'; do
  rg -Fq "$anchor" "$header" || {
    printf '%s\n' "missing strict FP8 header anchor: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'fn validate_finite_fp8_region(' \
  "code == b'\\x7f' || code == b'\\xff'" \
  'raise NonFiniteFp8Code(' \
  'dtype == @numeric_contract.StorageDType::f8_e4m3_finite()' \
  'validate_scale_region(region, scratch, chunk_bytes, read)'; do
  rg -Fq "$anchor" "$payload" || {
    printf '%s\n' "missing finite FP8 payload anchor: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'strict finite E4M3 payload' \
  'second pass copies finite E4M3 into the final arena' \
  'rejects FP8 dtype aliases' \
  'rejects both finite E4M3 NaN codes' \
  'chunk bounded and reports exact element' \
  'strict positive finite scalar scale'; do
  rg -Fq "$anchor" "$test_file" || {
    printf '%s\n' "missing hostile FP8 payload coverage: $anchor" >&2
    exit 1
  }
done

if rg -n \
  'fp8.*(ready|Ready)|scheduler|kv/|@cuda\.|internal/cuda|runtime/descriptor|engine/device_step' \
  "$header" "$payload" "$errors" "$test_file" "$package"; then
  printf '%s\n' 'FP8 numeric materializer gained readiness or execution authority' >&2
  exit 1
fi

printf '%s\n' 'FP8 authenticated numeric-weight materialization boundary passed'
