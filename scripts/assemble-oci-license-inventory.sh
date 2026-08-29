#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL
umask 077

fail() {
  printf '%s\n' "LunaFlux OCI license inventory rejected: $1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail 'no SHA-256 utility is available'
  fi
}

[ "$#" -eq 1 ] || {
  printf '%s\n' \
    'usage: assemble-oci-license-inventory.sh ABSOLUTE_NEW_OUTPUT' >&2
  exit 2
}

output=$1
case "$output" in
  /*) ;;
  *) fail 'output path is not absolute' ;;
esac
case "$output" in
  /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    fail 'output path is not safe and canonical'
    ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] ||
  fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  fail 'output parent is not canonical'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
license=$repo_root/LICENSE
[ -f "$license" ] && [ ! -L "$license" ] ||
  fail 'repository LICENSE is not a regular non-symlink file'
[ "$(grep -F -x -c 'license = "Apache-2.0"' "$repo_root/moon.mod")" -eq 1 ] ||
  fail 'moon.mod does not declare exactly one Apache-2.0 license'
[ "$(sed -n '1p' "$license")" = '                                 Apache License' ] ||
  fail 'repository LICENSE is not the admitted Apache-2.0 text'
[ "$(sed -n '2p' "$license")" = '                           Version 2.0, January 2004' ] ||
  fail 'repository LICENSE version is not Apache-2.0'

stage=$(mktemp "$output_parent/.lunaflux-oci-license.XXXXXX") ||
  fail 'could not create output-adjacent stage'
cleanup() {
  chmod u+w "$stage" 2>/dev/null || true
  rm -f "$stage"
}
trap cleanup EXIT HUP INT TERM

license_digest=$(sha256_file "$license")
printf '%s\n' \
  "{\"schema\":\"lunaflux.oci-product-license.v1\",\"entries\":[{\"component\":\"vectie/lunaflux\",\"license\":\"Apache-2.0\",\"path\":\"/usr/share/licenses/lunaflux/LICENSE\",\"sha256\":\"$license_digest\"}]}" \
  > "$stage"
chmod 444 "$stage"
mv "$stage" "$output"
trap - EXIT HUP INT TERM

printf '%s\n' 'LunaFlux OCI product-license inventory assembled.'
