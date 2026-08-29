#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
interface="$package_dir/pkg.generated.mbti"
production_files=$(rg --files "$package_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')

[ -f "$interface" ] || {
  echo "numeric loader generated interface is missing" >&2
  exit 1
}

actual_imports=$(awk '
  /^import \{/ { if (seen == 0) { inside=1; seen=1; next } }
  inside && /^}/ { exit }
  inside { print }
' "$package_dir/moon.pkg" | sed -E 's/^[[:space:]]*"([^"]+)".*/\1/' | sort)
expected_imports=$(printf '%s\n' \
  moonbitlang/core/encoding/utf8 \
  moonbitlang/core/json \
  moonbitlang/core/strconv \
  moonbitlang/x/crypto \
  vectie/lunaflux/device \
  vectie/lunaflux/internal/json_guard \
  vectie/lunaflux/model/device_materialize \
  vectie/lunaflux/model/numeric_contract \
  vectie/lunaflux/model/plan \
  vectie/lunaflux/model/safetensors \
  vectie/lunaflux/model/safetensors_reader \
  vectie/lunaflux/model/spec \
  vectie/lunaflux/runtime/approved_fs | sort)
[ "$actual_imports" = "$expected_imports" ] || {
  echo "numeric loader production import allowlist changed" >&2
  printf 'actual:\n%s\n' "$actual_imports" >&2
  exit 1
}

source_opaque() {
  candidate=$1
  printf '%s\n' "$candidate" | rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | rg -q '^  priv (mut )?[a-z][A-Za-z0-9_]* :' || return 1
  if printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\(|^  (pub |pub\(all\) )?[a-z][A-Za-z0-9_]* :'; then
    return 1
  fi
}

mbti_opaque() {
  candidate=$1
  printf '%s\n' "$candidate" | rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | rg -F -q '  // private fields' || return 1
  if printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\(|^  [a-z][A-Za-z0-9_]* :'; then
    return 1
  fi
}

if ! source_opaque 'pub struct PositiveControl {
  priv value : Int
}'; then
  echo "source opacity helper rejects private record" >&2
  exit 1
fi
if source_opaque 'pub struct TupleControl(Int)'; then
  echo "source opacity helper accepts tuple representation" >&2
  exit 1
fi
if source_opaque 'pub struct PublicControl {
  value : Int
}'; then
  echo "source opacity helper accepts public field" >&2
  exit 1
fi
if ! mbti_opaque 'pub struct PositiveControl {
  // private fields
}'; then
  echo "MBTI opacity helper rejects private fields" >&2
  exit 1
fi
if mbti_opaque 'pub struct TupleControl(Int)'; then
  echo "MBTI opacity helper accepts tuple representation" >&2
  exit 1
fi
if mbti_opaque 'pub struct PublicControl {
  value : Int
}'; then
  echo "MBTI opacity helper accepts public field" >&2
  exit 1
fi

for type in \
  NumericWeightArtifactDigest NumericWeightFileLimits \
  NumericWeightFileInspection NumericDeviceWeights \
  AuthenticatedNumericWeightAuthority FailedNumericDeviceWeightPreparation; do
  source=$(sed -n "/^pub struct ${type} {$/,/^}/p" $production_files)
  source_opaque "$source" || {
    echo "numeric loader source type is not opaque: $type" >&2
    exit 1
  }
  generated=$(sed -n "/^pub struct ${type} {$/,/^}/p" "$interface")
  mbti_opaque "$generated" || {
    echo "numeric loader MBTI type is not opaque: $type" >&2
    exit 1
  }
done

[ "$(rg --no-filename -F -x '#valtype' $production_files | wc -l | tr -d ' ')" = 1 ] || {
  echo "numeric loader value-type marker allowlist changed" >&2
  exit 1
}
rg -U -q '^#valtype\npub struct NumericWeightArtifactDigest \{' \
  "$package_dir/types.mbt" || {
  echo "numeric artifact digest lost adjacent value-type opacity" >&2
  exit 1
}

source_allocation_methods=$(rg --no-filename -o \
  '^pub fn [A-Za-z][A-Za-z0-9_]*::allocation\(' $production_files || true)
[ "$source_allocation_methods" = \
  'pub fn AuthenticatedNumericWeightAuthority::allocation(' ] || {
  echo "source allocation method allowlist changed or became empty" >&2
  printf '%s\n' "$source_allocation_methods" >&2
  exit 1
}
source_allocation_return_count=$(rg -U --no-filename -o \
  '^pub fn [A-Za-z][A-Za-z0-9_:]*\([^{}]*\) -> @device\.Allocation' \
  $production_files | rg -c '^pub fn ' || true)
[ "$source_allocation_return_count" = 1 ] || {
  echo "source allocation-return authority count changed" >&2
  exit 1
}
mbti_allocation_methods=$(rg --no-filename -o \
  '^pub fn [A-Za-z][A-Za-z0-9_]*::allocation\(' "$interface" || true)
[ "$mbti_allocation_methods" = \
  'pub fn AuthenticatedNumericWeightAuthority::allocation(' ] || {
  echo "MBTI allocation method allowlist changed or became empty" >&2
  printf '%s\n' "$mbti_allocation_methods" >&2
  exit 1
}

mbti_allocation_returns=$(rg --no-filename \
  '^pub fn .* -> @device\.Allocation( raise .*)?$' "$interface" || true)
[ "$mbti_allocation_returns" = \
  'pub fn AuthenticatedNumericWeightAuthority::allocation(Self) -> @device.Allocation raise NumericWeightFileError' ] || {
  echo "MBTI allocation-return authority allowlist changed" >&2
  printf '%s\n' "$mbti_allocation_returns" >&2
  exit 1
}
mbti_authority_factories=$(rg --no-filename \
  '^pub fn .* -> AuthenticatedNumericWeightAuthority( raise .*)?$' "$interface" || true)
[ "$mbti_authority_factories" = \
  'pub fn NumericDeviceWeights::authenticate(Self, expected_identity~ : @spec.ModelIdentity, expected_schema_digest~ : @numeric_contract.ModelNumericSchemaDigest, expected_layout~ : @device_materialize.DeviceWeightLayout, expected_artifact_digest~ : NumericWeightArtifactDigest) -> AuthenticatedNumericWeightAuthority raise NumericWeightFileError' ] || {
  echo "MBTI authenticated-authority factory allowlist changed" >&2
  printf '%s\n' "$mbti_authority_factories" >&2
  exit 1
}
source_authority_return_count=$(rg -U --no-filename -o \
  '^pub fn [A-Za-z][A-Za-z0-9_:]*\([^{}]*\) -> AuthenticatedNumericWeightAuthority' \
  $production_files | rg -c '^pub fn ' || true)
[ "$source_authority_return_count" = 1 ] || {
  echo "source authenticated-authority factory count changed" >&2
  exit 1
}

[ "$(rg --no-filename -F -o '{ weights: self }' $production_files | wc -l | tr -d ' ')" = 1 ] || {
  echo "authenticated authority factory count is not exactly one" >&2
  exit 1
}
rg -U -q 'pub fn NumericDeviceWeights::authenticate\([\s\S]*validate_numeric_weight_authority' \
  "$package_dir/types.mbt" || {
  echo "authority factory lost exact liveness join" >&2
  exit 1
}
rg -U -q 'pub fn AuthenticatedNumericWeightAuthority::allocation\([\s\S]*require_numeric_weights_open' \
  "$package_dir/types.mbt" || {
  echo "authenticated allocation access lost liveness validation" >&2
  exit 1
}

for required in \
  'lunaflux.numeric-safetensors.v1' \
  'lunaflux.tensor.v1.' \
  'NUMERIC_WEIGHT_MIN_CHUNK_BYTES' \
  'preflight_numeric_weight_scalars' \
  'F8_E4M3' \
  'NonFiniteFp8Code' \
  'validate_finite_fp8_region' \
  'ReservedI8Code' \
  'ZeroRowScaleMismatch' \
  'ArtifactDigestMismatch' \
  'AuthenticatedNumericWeightAuthority'; do
  rg -F -q "$required" $production_files || {
    echo "numeric loader invariant is missing: $required" >&2
    exit 1
  }
done

if rg -n 'lunaflux/(engine/|internal/cuda|model/llama)' "$package_dir/moon.pkg"; then
  echo "numeric loader crossed backend, engine, or model-family boundary" >&2
  exit 1
fi

echo "numeric weight loader boundary: ok"
