#!/bin/sh
set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/qwen3-capacity-receipt-common.sh"

qcap_fail() {
  printf 'Qwen3 capacity materialization rejected: %s\n' "$1" >&2
  exit 1
}

[ "$#" -eq 5 ] || qcap_fail \
  'usage: materialize-qwen3-authenticated-capacity.sh RELEASE_BIND_STDOUT#sha256=HEX DEPLOYMENT_ROOT#sha256=LAUNCH_SHA256 RUNTIME#sha256=HEX TOKEN_ID_BRIDGE#sha256=HEX ABSOLUTE_NEW_RECEIPT'

qcap_admit_inputs "$1" "$2" "$3" "$4"
output=$5
case "$output" in /*) ;; *) qcap_fail 'receipt output is not absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  qcap_fail 'receipt output path is unsafe'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] ||
  qcap_fail 'receipt output already exists'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  qcap_fail 'receipt output parent is not canonical'

scratch=$(mktemp -d "$output_parent/.qwen3-capacity.XXXXXX") ||
  qcap_fail 'could not create receipt scratch'
cleanup() {
  chmod -R u+rwX "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
}
trap cleanup EXIT HUP INT TERM
qcap_write_authentication "$scratch/authentication.v1"
authentication_sha=$(qcap_sha256_file "$scratch/authentication.v1")
qcap_write_receipt "$scratch/receipt.json" "$authentication_sha"
chmod 0444 "$scratch/receipt.json"
ln "$scratch/receipt.json" "$output" || qcap_fail 'receipt publication failed'
receipt_sha=$(qcap_sha256_file "$output")
rm -f -- "$scratch/receipt.json"
trap - EXIT HUP INT TERM
cleanup
printf '%s\n' 'schema=lunaflux-qwen3-capacity-materialization.v1'
printf 'receipt=%s#sha256=%s\n' "$output" "$receipt_sha"
printf 'configuration_sha256=%s\n' "$qcap_launch_sha"
printf 'authentication_sha256=%s\n' "$authentication_sha"
printf '%s\n' 'max_concurrency=32'
printf '%s\n' 'runtime_authority=0'
printf '%s\n' 'device_opened=0'
