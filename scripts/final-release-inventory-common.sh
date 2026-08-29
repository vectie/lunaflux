#!/bin/sh

# Shared fail-closed primitives for the Phase 9 final release inventory. This
# file is sourced by the assembler and verifier; it grants no release approval.

final_inventory_fail() {
  printf '%s\n' "LunaFlux final release inventory rejected: $1" >&2
  exit 1
}

final_inventory_is_sha256() {
  [ "${#1}" -eq 64 ] || return 1
  case "$1" in
    *[!0-9a-f]*|0000000000000000000000000000000000000000000000000000000000000000)
      return 1
      ;;
  esac
}

final_inventory_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    final_inventory_fail 'no SHA-256 utility is available'
  fi
}

final_inventory_mode() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

final_inventory_links() {
  if stat -f '%l' "$1" >/dev/null 2>&1; then
    stat -f '%l' "$1"
  else
    stat -c '%h' "$1"
  fi
}

final_inventory_require_file() {
  fir_path=$1
  fir_label=$2
  case "$fir_path" in
    /*) ;;
    *) final_inventory_fail "$fir_label path is not absolute" ;;
  esac
  case "$fir_path" in
    *//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
      final_inventory_fail "$fir_label path is not canonical"
      ;;
  esac
  [ -f "$fir_path" ] && [ ! -L "$fir_path" ] ||
    final_inventory_fail "$fir_label is not a regular non-symlink file"
  [ "$(final_inventory_links "$fir_path")" = 1 ] ||
    final_inventory_fail "$fir_label has a hard-link alias"
  fir_parent=$(CDPATH= cd -- "$(dirname -- "$fir_path")" && pwd -P)
  [ "$fir_parent/$(basename -- "$fir_path")" = "$fir_path" ] ||
    final_inventory_fail "$fir_label path contains a directory alias"
}

final_inventory_require_directory() {
  fir_path=$1
  case "$fir_path" in
    /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
      final_inventory_fail 'inventory root is not a safe canonical absolute path'
      ;;
    /*) ;;
    *) final_inventory_fail 'inventory root is not absolute' ;;
  esac
  [ -d "$fir_path" ] && [ ! -L "$fir_path" ] ||
    final_inventory_fail 'inventory root is not a non-symlink directory'
  [ "$(CDPATH= cd -- "$fir_path" && pwd -P)" = "$fir_path" ] ||
    final_inventory_fail 'inventory root contains a directory alias'
}

final_inventory_require_digest() {
  fir_path=$1
  fir_digest=$2
  fir_label=$3
  final_inventory_is_sha256 "$fir_digest" ||
    final_inventory_fail "$fir_label digest is invalid"
  [ "$(final_inventory_sha256 "$fir_path")" = "$fir_digest" ] ||
    final_inventory_fail "$fir_label digest does not match its bytes"
}

final_inventory_require_size() {
  fir_path=$1
  fir_maximum=$2
  fir_label=$3
  fir_size=$(wc -c < "$fir_path" | tr -d ' ')
  [ "$fir_size" -gt 0 ] && [ "$fir_size" -le "$fir_maximum" ] ||
    final_inventory_fail "$fir_label size is outside its fixed bound"
}

final_inventory_require_newline() {
  fir_path=$1
  fir_label=$2
  [ -s "$fir_path" ] || final_inventory_fail "$fir_label is empty"
  fir_byte=$(tail -c 1 "$fir_path" | od -An -tu1 | tr -d ' \n')
  [ "$fir_byte" = 10 ] ||
    final_inventory_fail "$fir_label is not newline terminated"
}

final_inventory_field() {
  fir_file=$1
  fir_line=$2
  fir_key=$3
  fir_value=$(sed -n "${fir_line}p" "$fir_file")
  case "$fir_value" in
    "$fir_key="*) printf '%s\n' "${fir_value#*=}" ;;
    *) final_inventory_fail "noncanonical field $fir_key" ;;
  esac
}

final_inventory_validate_line() {
  fir_line=$1
  fir_digest=${fir_line%%  *}
  fir_relative=${fir_line#*  }
  [ "$fir_relative" != "$fir_line" ] ||
    final_inventory_fail 'malformed file-inventory line'
  final_inventory_is_sha256 "$fir_digest" ||
    final_inventory_fail 'file inventory contains an invalid digest'
  case "$fir_relative" in
    ''|/*|*/|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
      final_inventory_fail 'file inventory contains an unsafe path'
      ;;
  esac
  [ "$fir_line" = "$fir_digest  $fir_relative" ] ||
    final_inventory_fail 'file inventory line is not canonical'
}

final_inventory_validate_files() {
  fir_root=$1
  fir_inventory=$2
  fir_expected_paths=$3
  final_inventory_require_newline "$fir_inventory" 'file inventory'
  while IFS= read -r fir_line; do
    final_inventory_validate_line "$fir_line"
    fir_digest=${fir_line%%  *}
    fir_relative=${fir_line#*  }
    [ -f "$fir_root/$fir_relative" ] && [ ! -L "$fir_root/$fir_relative" ] ||
      final_inventory_fail "inventoried payload is not a regular file: $fir_relative"
    [ "$(final_inventory_links "$fir_root/$fir_relative")" = 1 ] ||
      final_inventory_fail "inventoried payload has a hard-link alias: $fir_relative"
    [ "$(final_inventory_sha256 "$fir_root/$fir_relative")" = "$fir_digest" ] ||
      final_inventory_fail "inventoried payload digest mismatch: $fir_relative"
  done < "$fir_inventory"
  fir_actual=${FINAL_INVENTORY_SCRATCH:?}/declared.$$
  sed -n 's/^[0-9a-f][0-9a-f]*  //p' "$fir_inventory" > "$fir_actual"
  cmp -s "$fir_actual" "$fir_expected_paths" || {
    rm -f "$fir_actual"
    final_inventory_fail 'file inventory is not the exact expected path set'
  }
  rm -f "$fir_actual"
}

final_inventory_write_paths() {
  fir_root=$1
  fir_paths=$2
  fir_output=$3
  : > "$fir_output"
  while IFS= read -r fir_relative; do
    printf '%s  %s\n' \
      "$(final_inventory_sha256 "$fir_root/$fir_relative")" "$fir_relative" \
      >> "$fir_output"
  done < "$fir_paths"
}

final_inventory_require_image() {
  fir_image=$1
  case "$fir_image" in *@sha256:*) ;; *) final_inventory_fail 'OCI image is not digest pinned' ;; esac
  fir_name=${fir_image%@sha256:*}
  fir_digest=${fir_image##*@sha256:}
  [ -n "$fir_name" ] || final_inventory_fail 'OCI image repository is empty'
  case "$fir_name" in
    *@*|*://*|*[!A-Za-z0-9._:/-]*) final_inventory_fail 'OCI image repository is malformed' ;;
  esac
  case "${fir_name##*/}" in *:*) final_inventory_fail 'OCI image includes a mutable tag' ;; esac
  final_inventory_is_sha256 "$fir_digest" ||
    final_inventory_fail 'OCI image digest is invalid'
}
