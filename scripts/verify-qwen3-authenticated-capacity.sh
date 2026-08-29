#!/bin/sh
set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/qwen3-capacity-receipt-common.sh"

qcap_fail() {
  printf 'Qwen3 capacity verification rejected: %s\n' "$1" >&2
  exit 1
}

[ "$#" -eq 5 ] || qcap_fail \
  'usage: verify-qwen3-authenticated-capacity.sh RELEASE_BIND_STDOUT#sha256=HEX DEPLOYMENT_ROOT#sha256=LAUNCH_SHA256 RUNTIME#sha256=HEX TOKEN_ID_BRIDGE#sha256=HEX RECEIPT#sha256=HEX'

qcap_admit_inputs "$1" "$2" "$3" "$4"
qcap_require_pinned_file "$5" 'Qwen capacity receipt'
receipt=$qcap_pinned_path
receipt_sha=$qcap_pinned_sha
scratch=$(mktemp -d /tmp/lunaflux-qwen3-capacity-verify.XXXXXX) ||
  qcap_fail 'could not create verifier scratch'
trap 'rm -rf -- "$scratch"' EXIT HUP INT TERM
qcap_write_authentication "$scratch/authentication.v1"
authentication_sha=$(qcap_sha256_file "$scratch/authentication.v1")
qcap_write_receipt "$scratch/expected.json" "$authentication_sha"
cmp -s "$receipt" "$scratch/expected.json" ||
  qcap_fail 'capacity receipt differs from exact authenticated c32 inputs'
printf '%s\n' 'schema=lunaflux-qwen3-capacity-verification.v1'
printf 'receipt_sha256=%s\n' "$receipt_sha"
printf 'configuration_sha256=%s\n' "$qcap_launch_sha"
printf 'authentication_sha256=%s\n' "$authentication_sha"
printf '%s\n' 'max_concurrency=32'
printf '%s\n' 'outcome=passed'
