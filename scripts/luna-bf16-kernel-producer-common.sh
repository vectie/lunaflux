#!/bin/sh

# Shared validation for the offline BF16 kernel-set producer. This file is
# sourced by producer scripts and intentionally owns no ambient configuration.

lbf_fail() {
  printf 'Luna BF16 kernel producer failed: %s\n' "$1" >&2
  exit 1
}

lbf_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    lbf_fail 'sha256sum or shasum is required'
  fi
}

lbf_is_sha256() {
  printf '%s\n' "$1" |
    awk 'length != 64 || $0 !~ /^[0-9a-f]+$/ { exit 1 }'
}

lbf_is_key() {
  printf '%s\n' "$1" |
    awk '$0 !~ /^[a-z][a-z0-9_]*$/ || length > 64 { exit 1 }'
}

lbf_is_uint() {
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
}

lbf_is_target() {
  case "$1" in sm_[1-9][0-9]|sm_[1-9][0-9][0-9]) return 0 ;; *) return 1 ;; esac
}

lbf_is_family() {
  case "$1" in
    embedding_lookup|rms_norm|qk_rms_norm|positioned_rotary|residual_add|qkv_projection|dense_projection|gated_mlp|language_model_head|paged_attention)
      return 0
      ;;
    *) return 1 ;;
  esac
}

lbf_require_absolute_file() {
  lbf_path=$1
  case "$lbf_path" in /*) ;; *) lbf_fail "file path is not absolute: $lbf_path" ;; esac
  case "$lbf_path" in *//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    lbf_fail "file path is not canonical: $lbf_path"
    ;;
  esac
  [ -f "$lbf_path" ] && [ ! -L "$lbf_path" ] ||
    lbf_fail "input is not a regular non-symlink file: $lbf_path"
  lbf_parent=$(CDPATH= cd -- "$(dirname -- "$lbf_path")" && pwd -P)
  [ "$lbf_parent/$(basename -- "$lbf_path")" = "$lbf_path" ] ||
    lbf_fail "file path contains an alias or symlink: $lbf_path"
}

lbf_require_absolute_directory() {
  lbf_path=$1
  case "$lbf_path" in /*) ;; *) lbf_fail "directory path is not absolute: $lbf_path" ;; esac
  case "$lbf_path" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
    lbf_fail "directory path is not a safe canonical root: $lbf_path"
    ;;
  esac
  [ -d "$lbf_path" ] && [ ! -L "$lbf_path" ] ||
    lbf_fail "input is not a non-symlink directory: $lbf_path"
  [ "$(CDPATH= cd -- "$lbf_path" && pwd -P)" = "$lbf_path" ] ||
    lbf_fail "directory path contains an alias or symlink: $lbf_path"
}

lbf_require_newline() {
  [ -s "$1" ] || lbf_fail "$2 is empty"
  lbf_last=$(tail -c 1 "$1" | od -An -tu1 | tr -d ' \n')
  [ "$lbf_last" = 10 ] || lbf_fail "$2 is not newline terminated"
  if grep -q "$(printf '\r')" "$1"; then
    lbf_fail "$2 contains carriage returns"
  fi
}

lbf_inventory_paths() {
  sed -n 's/^[0-9a-f][0-9a-f]*  //p' "$1"
}

lbf_validate_inventory_line() {
  lbf_line=$1
  lbf_digest=${lbf_line%%  *}
  lbf_relative=${lbf_line#*  }
  [ "$lbf_relative" != "$lbf_line" ] || lbf_fail 'malformed inventory line'
  lbf_is_sha256 "$lbf_digest" || lbf_fail 'inventory digest is invalid'
  case "$lbf_relative" in
    ''|/*|*/|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
      lbf_fail "inventory path is invalid: $lbf_relative"
      ;;
  esac
  [ "$lbf_line" = "$lbf_digest  $lbf_relative" ] ||
    lbf_fail 'inventory line is not canonical'
}

lbf_validate_candidate_inventory() {
  lbf_inventory=$1
  lbf_root=$2
  lbf_scratch=$3
  lbf_require_newline "$lbf_inventory" 'candidate inventory'
  lbf_paths=$lbf_scratch/candidate.paths
  lbf_sorted=$lbf_scratch/candidate.sorted
  lbf_inventory_paths "$lbf_inventory" >"$lbf_paths"
  LC_ALL=C sort "$lbf_paths" >"$lbf_sorted"
  cmp -s "$lbf_paths" "$lbf_sorted" || lbf_fail 'candidate inventory is not sorted'
  [ -z "$(uniq -d "$lbf_paths" | sed -n '1p')" ] ||
    lbf_fail 'candidate inventory contains duplicate paths'
  while IFS= read -r lbf_line; do
    lbf_validate_inventory_line "$lbf_line"
    lbf_digest=${lbf_line%%  *}
    lbf_relative=${lbf_line#*  }
    case "$lbf_relative" in
      candidate-set.v1|sources/*.cu|recipes/*.recipe) ;;
      *) lbf_fail "unsupported candidate payload: $lbf_relative" ;;
    esac
    lbf_source=$lbf_root/$lbf_relative
    [ -f "$lbf_source" ] && [ ! -L "$lbf_source" ] ||
      lbf_fail "candidate payload is not a regular file: $lbf_relative"
    [ "$(lbf_sha256_file "$lbf_source")" = "$lbf_digest" ] ||
      lbf_fail "candidate payload digest mismatch: $lbf_relative"
  done <"$lbf_inventory"
  [ "$(grep -c '^.*  candidate-set\.v1$' "$lbf_inventory")" -eq 1 ] ||
    lbf_fail 'candidate inventory must contain candidate-set.v1 exactly once'
  if find "$lbf_root" -type l -print | grep -q .; then
    lbf_fail 'candidate root contains a symbolic link'
  fi
  if find "$lbf_root" ! -type d ! -type f -print | grep -q .; then
    lbf_fail 'candidate root contains a special filesystem object'
  fi
  lbf_actual=$lbf_scratch/candidate.actual
  find "$lbf_root" -type f -print | sed "s#^$lbf_root/##" | LC_ALL=C sort >"$lbf_actual"
  cmp -s "$lbf_actual" "$lbf_paths" ||
    lbf_fail 'candidate inventory is not the exact candidate-root file set'
}

lbf_recipe_value() {
  lbf_key=$1
  lbf_recipe=$2
  [ "$(grep -c "^${lbf_key}=" "$lbf_recipe")" -eq 1 ] ||
    lbf_fail "recipe must contain exactly one $lbf_key field"
  sed -n "s/^${lbf_key}=//p" "$lbf_recipe"
}

lbf_write_inventory() {
  lbf_root=$1
  lbf_output=$2
  lbf_output_relative=${lbf_output#"$lbf_root"/}
  find "$lbf_root" -type f -print | sed "s#^$lbf_root/##" | LC_ALL=C sort |
    while IFS= read -r lbf_relative; do
      [ "$lbf_relative" = "$lbf_output_relative" ] && continue
      printf '%s  %s\n' "$(lbf_sha256_file "$lbf_root/$lbf_relative")" "$lbf_relative"
    done >"$lbf_output"
}
