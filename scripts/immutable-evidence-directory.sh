#!/usr/bin/env bash

# Generic filesystem mechanics for immutable evidence directories. Callers own
# every evidence schema, result field, success decision, and admission claim.

lunaflux_evidence_manifest_sha256=

lunaflux_evidence_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

lunaflux_prepare_evidence_manifest() {
  local evidence_dir=$1
  local files_manifest=$evidence_dir/FILES.sha256
  local paths paths_sorted manifest_status file relative file_sha

  chmod -R u+rwX "$evidence_dir" 2>/dev/null || true
  : >"$files_manifest" || return 1
  paths=$(mktemp) || return 1
  paths_sorted=$(mktemp) || {
    rm -f -- "$paths"
    return 1
  }
  if ! find "$evidence_dir" -type f ! -path "$files_manifest" \
    ! -path "$evidence_dir/RESULT.txt" -print0 >"$paths"; then
    rm -f -- "$paths" "$paths_sorted"
    return 1
  fi
  if ! LC_ALL=C sort -z "$paths" >"$paths_sorted"; then
    rm -f -- "$paths" "$paths_sorted"
    return 1
  fi
  manifest_status=0
  while IFS= read -r -d '' file; do
    relative=${file#"$evidence_dir"/}
    file_sha=$(lunaflux_evidence_sha256_file "$file") || {
      manifest_status=1
      break
    }
    printf '%s  %s\n' "$file_sha" "$relative" >>"$files_manifest" || {
      manifest_status=1
      break
    }
  done <"$paths_sorted"
  rm -f -- "$paths" "$paths_sorted"
  [ "$manifest_status" -eq 0 ] || return 1
  lunaflux_evidence_manifest_sha256=$(
    lunaflux_evidence_sha256_file "$files_manifest"
  ) || return 1
}

lunaflux_seal_evidence_directory() {
  local evidence_dir=$1
  local executable
  shift

  for executable in "$@"; do
    case "$executable" in
      "$evidence_dir"/*) ;;
      *) return 1 ;;
    esac
    [ -f "$executable" ] && [ ! -L "$executable" ] || return 1
    [ "$(realpath -- "$executable")" = "$executable" ] || return 1
  done
  find "$evidence_dir" -type f -exec chmod 0444 {} + || return 1
  for executable in "$@"; do
    chmod 0555 "$executable" || return 1
  done
  find "$evidence_dir" -type d -exec chmod 0555 {} + || return 1
}
