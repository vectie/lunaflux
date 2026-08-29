#!/bin/sh

# Shared fail-closed parsing for the offline deployment-bundle assembler and
# verifier. This file is sourced only by those scripts.

bundle_fail() {
  printf '%s\n' "LunaFlux deployment bundle rejected: $1" >&2
  exit 1
}

bundle_is_lower_sha256() {
  [ "${#1}" -eq 64 ] || return 1
  case "$1" in
    *[!0-9a-f]*|0000000000000000000000000000000000000000000000000000000000000000)
      return 1
      ;;
  esac
}

bundle_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    bundle_fail 'no SHA-256 utility is available'
  fi
}

bundle_file_mode() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

bundle_file_owner_uid() {
  if stat -f '%u' "$1" >/dev/null 2>&1; then
    stat -f '%u' "$1"
  else
    stat -c '%u' "$1"
  fi
}

bundle_file_link_count() {
  if stat -f '%l' "$1" >/dev/null 2>&1; then
    stat -f '%l' "$1"
  else
    stat -c '%h' "$1"
  fi
}

bundle_require_mode() {
  brc_actual_mode=$(bundle_file_mode "$1")
  [ "$brc_actual_mode" = "$2" ] ||
    bundle_fail "invalid mode $brc_actual_mode for $1; expected $2"
}

bundle_is_strict_relative() {
  brc_value=$1
  case "$brc_value" in
    ''|/*|*/|*//*|*[!A-Za-z0-9._/-]*) return 1 ;;
  esac
  brc_old_ifs=$IFS
  IFS=/
  set -- $brc_value
  IFS=$brc_old_ifs
  for brc_component do
    case "$brc_component" in ''|.|..) return 1 ;; esac
  done
}

bundle_require_canonical_absolute_file() {
  brc_value=$1
  case "$brc_value" in
    /*) ;;
    *) bundle_fail "input file path is not absolute: $brc_value" ;;
  esac
  case "$brc_value" in
    *//*|*/./*|*/../*|*/.|*/..) bundle_fail "input file path is not canonical: $brc_value" ;;
    *[!A-Za-z0-9._/-]*) bundle_fail "input file path has unsupported characters: $brc_value" ;;
  esac
  [ -f "$brc_value" ] && [ ! -L "$brc_value" ] ||
    bundle_fail "input is not a regular non-symlink file: $brc_value"
  brc_parent=$(CDPATH= cd -- "$(dirname -- "$brc_value")" && pwd -P)
  [ "$brc_parent/$(basename -- "$brc_value")" = "$brc_value" ] ||
    bundle_fail "input file path contains an alias or symlink: $brc_value"
}

bundle_require_canonical_absolute_directory() {
  brc_value=$1
  case "$brc_value" in
    /*) ;;
    *) bundle_fail "input directory path is not absolute: $brc_value" ;;
  esac
  case "$brc_value" in
    /|*//*|*/./*|*/../*|*/.|*/..)
      bundle_fail "input directory path is not a safe canonical root: $brc_value"
      ;;
    *[!A-Za-z0-9._/-]*)
      bundle_fail "input directory path has unsupported characters: $brc_value"
      ;;
  esac
  [ -d "$brc_value" ] && [ ! -L "$brc_value" ] ||
    bundle_fail "input is not a non-symlink directory: $brc_value"
  [ "$(CDPATH= cd -- "$brc_value" && pwd -P)" = "$brc_value" ] ||
    bundle_fail "input directory path contains an alias or symlink: $brc_value"
}

bundle_require_digest() {
  brc_file=$1
  brc_expected=$2
  brc_label=$3
  bundle_is_lower_sha256 "$brc_expected" || bundle_fail "$brc_label digest is invalid"
  [ "$(bundle_sha256_file "$brc_file")" = "$brc_expected" ] ||
    bundle_fail "$brc_label digest does not match its bytes"
}

bundle_require_newline_terminated() {
  brc_file=$1
  brc_label=$2
  [ -s "$brc_file" ] || bundle_fail "$brc_label is empty"
  brc_last_byte=$(tail -c 1 "$brc_file" | od -An -tu1 | tr -d ' \n')
  [ "$brc_last_byte" = 10 ] || bundle_fail "$brc_label is not newline terminated"
}

bundle_inventory_paths() {
  sed -n 's/^[0-9a-f][0-9a-f]*  //p' "$1"
}

bundle_validate_inventory_line() {
  brc_line=$1
  brc_digest=${brc_line%%  *}
  brc_relative=${brc_line#*  }
  [ "$brc_relative" != "$brc_line" ] || bundle_fail 'malformed inventory line'
  bundle_is_lower_sha256 "$brc_digest" || bundle_fail 'invalid inventory digest'
  bundle_is_strict_relative "$brc_relative" ||
    bundle_fail "invalid inventory path: $brc_relative"
  [ "$brc_line" = "$brc_digest  $brc_relative" ] || bundle_fail 'noncanonical inventory line'
}

bundle_validate_inventory() {
  brc_inventory=$1
  brc_root=$2
  brc_kind=$3
  [ -s "$brc_inventory" ] || bundle_fail "$brc_kind inventory is empty"
  bundle_require_newline_terminated "$brc_inventory" "$brc_kind inventory"
  case "$brc_inventory" in "$brc_root"/*) bundle_fail "$brc_kind inventory is inside its payload root" ;; esac
  if grep -q "$(printf '\r')" "$brc_inventory"; then
    bundle_fail "$brc_kind inventory contains carriage returns"
  fi
  [ -n "${BUNDLE_SCRATCH_DIR:-}" ] && [ -d "$BUNDLE_SCRATCH_DIR" ] ||
    bundle_fail 'private verifier scratch is unavailable'
  brc_paths=$BUNDLE_SCRATCH_DIR/inventory.paths.$$
  brc_sorted=$BUNDLE_SCRATCH_DIR/inventory.sorted.$$
  bundle_inventory_paths "$brc_inventory" > "$brc_paths"
  LC_ALL=C sort "$brc_paths" > "$brc_sorted"
  cmp -s "$brc_paths" "$brc_sorted" || {
    rm -f "$brc_paths" "$brc_sorted"
    bundle_fail "$brc_kind inventory is not canonically sorted"
  }
  rm -f "$brc_paths" "$brc_sorted"
  brc_duplicate=$(bundle_inventory_paths "$brc_inventory" | uniq -d | sed -n '1p')
  [ -z "$brc_duplicate" ] || bundle_fail "$brc_kind inventory contains duplicate paths"
  while IFS= read -r brc_line; do
    bundle_validate_inventory_line "$brc_line"
    brc_digest=${brc_line%%  *}
    brc_relative=${brc_line#*  }
    brc_source=$brc_root/$brc_relative
    [ -f "$brc_source" ] && [ ! -L "$brc_source" ] ||
      bundle_fail "$brc_kind inventory entry is not a regular file: $brc_relative"
    [ "$(bundle_sha256_file "$brc_source")" = "$brc_digest" ] ||
      bundle_fail "$brc_kind payload digest mismatch: $brc_relative"
    case "$brc_kind:$brc_relative" in
      model:*.json|model:*.safetensors) ;;
      policy:*.json) ;;
      kernel:*.json|kernel:*.cubin|kernel:*.fatbin|kernel:*.bin) ;;
      library:*.so|library:*.so.[0-9]*) ;;
      *) bundle_fail "unsupported $brc_kind payload: $brc_relative" ;;
    esac
  done < "$brc_inventory"
  if find "$brc_root" -type l -print | grep -q .; then
    bundle_fail "$brc_kind root contains a symbolic link"
  fi
  if find "$brc_root" ! -type d ! -type f -print | grep -q .; then
    bundle_fail "$brc_kind root contains a special filesystem object"
  fi
  brc_actual=$BUNDLE_SCRATCH_DIR/inventory.actual.$$
  find "$brc_root" -type f -print | sed "s#^$brc_root/##" | LC_ALL=C sort > "$brc_actual"
  brc_declared=$BUNDLE_SCRATCH_DIR/inventory.declared.$$
  bundle_inventory_paths "$brc_inventory" > "$brc_declared"
  cmp -s "$brc_actual" "$brc_declared" || {
    rm -f "$brc_actual" "$brc_declared"
    bundle_fail "$brc_kind inventory is not the exact source-root file set"
  }
  rm -f "$brc_actual" "$brc_declared"
}

bundle_copy_inventory() {
  brc_inventory=$1
  brc_source_root=$2
  brc_target_root=$3
  while IFS= read -r brc_line; do
    brc_relative=${brc_line#*  }
    brc_target=$brc_target_root/$brc_relative
    mkdir -p "$(dirname -- "$brc_target")"
    cp "$brc_source_root/$brc_relative" "$brc_target"
    chmod 444 "$brc_target"
  done < "$brc_inventory"
}

bundle_write_file_inventory() {
  brc_root=$1
  brc_output=$2
  brc_output_relative=${brc_output#"$brc_root"/}
  find "$brc_root" -type f -print | sed "s#^$brc_root/##" | LC_ALL=C sort |
    while IFS= read -r brc_relative; do
      [ "$brc_relative" = "$brc_output_relative" ] && continue
      printf '%s  %s\n' "$(bundle_sha256_file "$brc_root/$brc_relative")" "$brc_relative"
    done > "$brc_output"
}

bundle_evidence_value() {
  brc_evidence=$1
  brc_line_number=$2
  brc_key=$3
  brc_line=$(sed -n "${brc_line_number}p" "$brc_evidence")
  case "$brc_line" in
    "$brc_key="*) printf '%s\n' "${brc_line#*=}" ;;
    *) bundle_fail "noncanonical evidence field $brc_key" ;;
  esac
}
