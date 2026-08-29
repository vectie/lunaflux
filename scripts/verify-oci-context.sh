#!/bin/sh

set -eu

fail() {
  printf '%s\n' "LunaFlux OCI context rejected: $1" >&2
  exit 1
}

is_lower_sha256() {
  [ "${#1}" -eq 64 ] || return 1
  case "$1" in
    *[!0-9a-f]*) return 1 ;;
  esac
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

file_mode() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

file_link_count() {
  if stat -f '%l' "$1" >/dev/null 2>&1; then
    stat -f '%l' "$1"
  else
    stat -c '%h' "$1"
  fi
}

require_mode() {
  actual_mode=$(file_mode "$1")
  [ "$actual_mode" = "$2" ] ||
    fail "invalid mode $actual_mode for $1; expected $2"
}

require_single_line() {
  [ -f "$1" ] || fail "missing metadata file $1"
  [ "$(wc -l < "$1" | tr -d ' ')" -eq 1 ] ||
    fail "metadata must contain exactly one line: $1"
}

require_nonempty_file() {
  [ -s "$1" ] || fail "missing or empty metadata file $1"
}

require_strict_relative() {
  case "$1" in
    ''|/*|*..*|*[!A-Za-z0-9._/-]*)
      fail "invalid strict relative path: $1"
      ;;
  esac
}

require_elf_architecture() {
  file=$1
  architecture=$2
  magic=$(od -An -tx1 -N4 "$file" | tr -d ' \n')
  [ "$magic" = '7f454c46' ] || fail "non-ELF Linux payload: $file"
  class_data=$(od -An -tu1 -j4 -N2 "$file" |
    awk 'NF >= 2 {print $1 ":" $2}')
  [ "$class_data" = '2:1' ] || fail "payload is not ELF64 little-endian: $file"
  machine=$(od -An -tu1 -j18 -N2 "$file" |
    awk 'NF >= 2 {print $1 ":" $2}')
  case "$architecture:$machine" in
    x86_64:62:0|aarch64:183:0) ;;
    *) fail "ELF architecture mismatch for $file" ;;
  esac
}

manifest_has_exact_path_value() {
  manifest_file=$1
  expected_path=$2
  awk -v expected_path="$expected_path" '
    BEGIN {
      state = 0
      in_string = 0
      escaped = 0
      found = 0
      document = ""
    }
    {
      if (NR > 1) document = document "\n"
      document = document $0
    }
    END {
      for (position = 1; position <= length(document); position += 1) {
        character = substr(document, position, 1)
        if (in_string) {
          if (escaped) {
            escaped = 0
            token = token character
          } else if (character == "\\") {
            escaped = 1
            had_escape = 1
          } else if (character == "\"") {
            in_string = 0
            if (role == 2 && !had_escape && token == expected_path) {
              found = 1
            }
            if (role == 0 && !had_escape && token == "path") {
              state = 1
            } else {
              state = 0
            }
          } else {
            token = token character
          }
        } else if (character == "\"") {
          in_string = 1
          escaped = 0
          had_escape = 0
          token = ""
          role = state
        } else if (character ~ /[ \t\r\n]/) {
          continue
        } else if (state == 1 && character == ":") {
          state = 2
        } else {
          state = 0
        }
      }
      exit(found ? 0 : 1)
    }
  ' "$manifest_file"
}

[ "$#" -eq 2 ] || {
  printf '%s\n' 'usage: verify-oci-context.sh BASE_IMAGE CONTEXT_DIR' >&2
  exit 2
}

base_image=$1
context_dir=$2
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

case "$base_image" in
  *@sha256:*) ;;
  *) fail 'BASE_IMAGE is not digest pinned' ;;
esac
base_name=${base_image%@sha256:*}
base_digest=${base_image##*@sha256:}
[ -n "$base_name" ] || fail 'BASE_IMAGE repository is empty'
case "$base_name" in
  *@*|*://*|*[!A-Za-z0-9._:/-]*) fail 'BASE_IMAGE repository is malformed' ;;
esac
case "${base_name##*/}" in
  *:*) fail 'BASE_IMAGE includes a mutable tag' ;;
esac
is_lower_sha256 "$base_digest" ||
  fail 'BASE_IMAGE digest is not 64 lowercase hexadecimal characters'

[ -d "$context_dir" ] || fail 'context directory does not exist'
context_dir=$(CDPATH= cd -- "$context_dir" && pwd -P)
rootfs=$context_dir/rootfs
metadata=$context_dir/metadata
[ -d "$rootfs" ] || fail 'rootfs directory is missing'
[ -d "$metadata" ] || fail 'metadata directory is missing'
require_mode "$rootfs" 555
require_mode "$metadata" 555

top_entries=$(find "$context_dir" -mindepth 1 -maxdepth 1 -print |
  sed "s#^$context_dir/##" | LC_ALL=C sort)
[ "$top_entries" = "metadata
rootfs" ] || fail 'context has unexpected top-level entries'
if find "$context_dir" -type l -print | grep -q .; then
  fail 'symbolic links are forbidden in the context'
fi
if find "$context_dir" ! -type d ! -type f -print | grep -q .; then
  fail 'context contains a non-regular filesystem object'
fi
if find "$context_dir" -mindepth 1 -name '*[!A-Za-z0-9._-]*' -print |
  grep -q .; then
  fail 'context path names must use only ASCII letters, digits, dot, underscore, and dash'
fi
find "$rootfs" "$metadata" -type f -print |
  while IFS= read -r context_file; do
    [ "$(file_link_count "$context_file")" = 1 ] ||
      fail "context file has a hard-link alias: $context_file"
  done

metadata_entries=$(find "$metadata" -mindepth 1 -maxdepth 1 -type f -print |
  sed "s#^$metadata/##" | LC_ALL=C sort)
[ "$metadata_entries" = "artifacts.sha256
base-image.ref
kernel-manifest.relative
kernel-manifest.sha256
license-inventory.json
linux-architecture
runtime-libraries.list" ] || fail 'metadata directory is not the exact control-file set'
if find "$metadata" -mindepth 1 ! -type f -print | grep -q .; then
  fail 'metadata contains an unexpected directory or object'
fi

for directory in $(find "$rootfs" "$metadata" -type d -print); do
  [ "$directory" = "$rootfs" ] || [ "$directory" = "$metadata" ] ||
    require_mode "$directory" 555
done

require_nonempty_file "$metadata/artifacts.sha256"
require_mode "$metadata/artifacts.sha256" 444
for name in base-image.ref kernel-manifest.relative kernel-manifest.sha256 \
  license-inventory.json linux-architecture runtime-libraries.list; do
  require_single_line "$metadata/$name"
  require_mode "$metadata/$name" 444
done

[ "$(sed -n '1p' "$metadata/base-image.ref")" = "$base_image" ] ||
  fail 'metadata base-image.ref does not match BASE_IMAGE'
architecture=$(sed -n '1p' "$metadata/linux-architecture")
case "$architecture" in
  x86_64|aarch64) ;;
  *) fail 'unsupported linux-architecture metadata' ;;
esac

for binary in lunaflux lunaflux-device-worker; do
  path=$rootfs/opt/lunaflux/bin/$binary
  [ -f "$path" ] || fail "missing runtime executable $binary"
  require_mode "$path" 555
  require_elf_architecture "$path" "$architecture"
done

mount_marker=$rootfs/var/lib/lunaflux/model/.mount-contract
[ -f "$mount_marker" ] || fail 'model read-only mount marker is missing'
require_mode "$mount_marker" 444
[ "$(sed -n '1p' "$mount_marker")" = 'external-read-only-model-root-required' ] ||
  fail 'model mount marker is invalid'

license_source=$repo_root/LICENSE
license_payload=$rootfs/usr/share/licenses/lunaflux/LICENSE
[ -f "$license_source" ] && [ ! -L "$license_source" ] ||
  fail 'repository LICENSE is unavailable'
[ -f "$license_payload" ] && [ ! -L "$license_payload" ] ||
  fail 'LunaFlux product license is missing from the rootfs'
require_mode "$license_payload" 444
cmp -s "$license_source" "$license_payload" ||
  fail 'rootfs product license does not match the repository LICENSE'
license_digest=$(sha256_file "$license_source")
expected_license_inventory="{\"schema\":\"lunaflux.oci-product-license.v1\",\"entries\":[{\"component\":\"vectie/lunaflux\",\"license\":\"Apache-2.0\",\"path\":\"/usr/share/licenses/lunaflux/LICENSE\",\"sha256\":\"$license_digest\"}]}"
[ "$(sed -n '1p' "$metadata/license-inventory.json")" = \
    "$expected_license_inventory" ] ||
  fail 'product license inventory is not the exact canonical record'

for directory in $(find "$rootfs" -mindepth 1 -type d -print); do
  relative=${directory#"$rootfs"/}
  require_strict_relative "$relative"
  case "$relative" in
    opt|opt/lunaflux|opt/lunaflux/bin|opt/lunaflux/lib|opt/lunaflux/kernels|opt/lunaflux/kernels/*|usr|usr/share|usr/share/licenses|usr/share/licenses/lunaflux|var|var/lib|var/lib/lunaflux|var/lib/lunaflux/model)
      ;;
    *) fail "unexpected runtime directory: $relative" ;;
  esac
  if [ "$relative" != opt/lunaflux/lib ] &&
    [ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    fail "empty runtime directory is forbidden: $relative"
  fi
done
for required_directory in opt/lunaflux/bin opt/lunaflux/lib \
  opt/lunaflux/kernels usr/share/licenses/lunaflux var/lib/lunaflux/model; do
  [ -d "$rootfs/$required_directory" ] ||
    fail "missing runtime directory: $required_directory"
done

for payload in $(find "$rootfs" -type f -print); do
  relative=${payload#"$rootfs"/}
  require_strict_relative "$relative"
  case "$relative" in
    opt/lunaflux/bin/lunaflux|opt/lunaflux/bin/lunaflux-device-worker)
      ;;
    opt/lunaflux/lib/*|opt/lunaflux/kernels/*|usr/share/licenses/lunaflux/LICENSE|var/lib/lunaflux/model/.mount-contract)
      require_mode "$payload" 444
      ;;
    *) fail "unexpected runtime payload: $relative" ;;
  esac
  case "$relative" in
    *python*|*pypy*|*nvrtc*|*/moon|*/moonc|*/clang*|*/gcc*|*/g++*|*/nvcc*|*/ptxas*|*/cmake*|*/ninja*|*/make|*/bash|*/sh|*.ptx|*.cu|*.cuh|*.ll|*.bc|*.o|*.a)
      fail "compiler, shell, source, or JIT payload is forbidden: $relative"
      ;;
  esac
done

if command -v strings >/dev/null 2>&1; then
  for runtime_file in "$rootfs/opt/lunaflux/bin/lunaflux" \
    "$rootfs/opt/lunaflux/bin/lunaflux-device-worker" \
    $(find "$rootfs/opt/lunaflux/lib" -type f -print 2>/dev/null); do
    if strings "$runtime_file" |
      grep -E -i 'libnvrtc|(^|[/ ])nvrtc([./ ]|$)|libpython|/python([0-9.]*)?$' \
      >/dev/null 2>&1; then
      fail "runtime payload references a JIT or Python runtime: $runtime_file"
    fi
  done
else
  fail 'strings is required for runtime dependency screening'
fi

libraries=$(sed -n '1p' "$metadata/runtime-libraries.list")
actual_libraries=$(find "$rootfs/opt/lunaflux/lib" -type f -print 2>/dev/null |
  sed "s#^$rootfs/##" | LC_ALL=C sort)
if [ "$libraries" = none ]; then
  [ -z "$actual_libraries" ] || fail 'runtime-libraries.list says none but libraries exist'
else
  declared_libraries=$(tr ',' '\n' < "$metadata/runtime-libraries.list" |
    LC_ALL=C sort)
  [ "$declared_libraries" = "$actual_libraries" ] ||
    fail 'runtime-libraries.list does not exactly match staged libraries'
  for library in $declared_libraries; do
    require_strict_relative "$library"
    case "$library" in
      opt/lunaflux/lib/*.so|opt/lunaflux/lib/*.so.[0-9]*) ;;
      *) fail "runtime library has an unsupported name: $library" ;;
    esac
    require_elf_architecture "$rootfs/$library" "$architecture"
  done
fi

manifest_relative=$(sed -n '1p' "$metadata/kernel-manifest.relative")
require_strict_relative "$manifest_relative"
case "$manifest_relative" in
  *.json) ;;
  *) fail 'kernel manifest must be a JSON descendant of the kernel root' ;;
esac
manifest=$rootfs/opt/lunaflux/kernels/$manifest_relative
[ -f "$manifest" ] || fail 'declared kernel manifest is missing'
manifest_digest=$(sed -n '1p' "$metadata/kernel-manifest.sha256")
is_lower_sha256 "$manifest_digest" || fail 'kernel manifest digest is invalid'
[ "$(sha256_file "$manifest")" = "$manifest_digest" ] ||
  fail 'kernel manifest digest does not match its bytes'
if grep -E -i 'nvrtc|runtime[_-]?jit|developer[_-]?jit|\.ptx|source[_-]?(path|code)' \
  "$manifest" >/dev/null 2>&1; then
  fail 'kernel manifest contains runtime compiler or JIT material'
fi

module_count=0
for module in $(find "$rootfs/opt/lunaflux/kernels" -type f -print); do
  [ "$module" = "$manifest" ] && continue
  module_count=$((module_count + 1))
  relative=${module#"$rootfs/opt/lunaflux/kernels"/}
  case "$relative" in
    *.cubin|*.fatbin|*.bin) ;;
    *) fail "non-AOT kernel payload is forbidden: $relative" ;;
  esac
  digest=$(sha256_file "$module")
  grep -F "$digest" "$manifest" >/dev/null 2>&1 ||
    fail "kernel manifest does not bind staged module: $relative"
  manifest_has_exact_path_value "$manifest" "$relative" ||
    fail "kernel manifest does not name exact staged module path: $relative"
done
[ "$module_count" -gt 0 ] || fail 'kernel root contains no AOT module'

checksums=$metadata/artifacts.sha256
declared_paths=$(sed -n 's/^[0-9a-f][0-9a-f]*  //p' "$checksums" | LC_ALL=C sort)
actual_paths=$(find "$rootfs" -type f -print | sed "s#^$rootfs/##" |
  LC_ALL=C sort)
[ "$declared_paths" = "$actual_paths" ] ||
  fail 'artifacts.sha256 is not the exact runtime payload inventory'
duplicates=$(printf '%s\n' "$declared_paths" | uniq -d)
[ -z "$duplicates" ] || fail 'artifacts.sha256 contains duplicate paths'
while IFS= read -r line; do
  digest=${line%%  *}
  relative=${line#*  }
  [ "$relative" != "$line" ] || fail 'malformed artifacts.sha256 line'
  is_lower_sha256 "$digest" || fail 'invalid digest in artifacts.sha256'
  require_strict_relative "$relative"
  [ -f "$rootfs/$relative" ] || fail "missing checksummed payload: $relative"
  [ "$(sha256_file "$rootfs/$relative")" = "$digest" ] ||
    fail "payload digest mismatch: $relative"
done < "$checksums"

printf '%s\n' 'LunaFlux OCI build context is exact and compiler/JIT-free.'
