#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

command_source=cmd/lunaflux/main.mbt
report_source=cmd/lunaflux/operator_doctor.mbt
test_source=cmd/lunaflux/operator_doctor_wbtest.mbt

for cli_anchor in \
  'lunaflux legacy-doctor --json' \
  '[_, "legacy-doctor", "--json"] => print_doctor_json()'
do
  if ! rg -F "$cli_anchor" "$command_source" >/dev/null; then
    echo "structured doctor CLI lost exact grammar: $cli_anchor" >&2
    exit 1
  fi
done

for authority_anchor in \
  'schema_version: "lunaflux.operator-doctor.v1"' \
  'readiness: false' \
  'physical_cuda_authority: false' \
  'external_tls_authority: false' \
  'external_auth_authority: false' \
  'status: "not_proven"'
do
  if ! rg -F "$authority_anchor" "$report_source" >/dev/null; then
    echo "structured doctor lost authority boundary: $authority_anchor" >&2
    exit 1
  fi
done

for hostile_anchor in \
  'structured doctor is versioned bounded and explicitly non-authoritative' \
  'structured doctor preserves stable unavailable reasons' \
  'human doctor remains aligned with the structured projection' \
  'root["readiness"] is False' \
  'root["physical_cuda_authority"] is False' \
  'root["external_tls_authority"] is False' \
  'root["external_auth_authority"] is False'
do
  if ! rg -F "$hostile_anchor" "$test_source" >/dev/null; then
    echo "structured doctor hostile coverage drifted: $hostile_anchor" >&2
    exit 1
  fi
done

if rg -ni \
  '(physical_cuda_authority[[:space:]]*:[[:space:]]*true|external_(tls|auth)_authority[[:space:]]*:[[:space:]]*true|readiness[[:space:]]*:[[:space:]]*true)' \
  "$report_source" "$test_source" >/dev/null; then
  echo "structured doctor fabricated release, TLS, auth, or readiness authority" >&2
  exit 1
fi

for file in "$report_source" "$test_source"
do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "structured doctor source exceeds size boundary: $file ($lines)" >&2
    exit 1
  fi
done

moon check cmd/lunaflux --target native --deny-warn
moon test cmd/lunaflux --target native --deny-warn

structured=$(moon run --target native cmd/lunaflux -- legacy-doctor --json)
[ "$(printf '%s\n' "$structured" | wc -l | tr -d ' ')" -eq 1 ] || {
  echo "structured doctor emitted a multi-line or ambient diagnostic" >&2
  exit 1
}
for output_anchor in \
  '"schema_version":"lunaflux.operator-doctor.v1"' \
  '"readiness":false' \
  '"physical_cuda_authority":false' \
  '"external_tls_authority":false' \
  '"external_auth_authority":false'
do
  if ! printf '%s\n' "$structured" | rg -F "$output_anchor" >/dev/null; then
    echo "structured doctor output lost exact field: $output_anchor" >&2
    exit 1
  fi
done

if printf '%s\n' "$structured" | rg -ni \
  '(model_root|kernel_root|credential|secret|private key|bearer)' >/dev/null; then
  echo "structured doctor leaked a deployment path or secret-shaped field" >&2
  exit 1
fi

echo "LunaFlux structured operator doctor boundary: ok"
