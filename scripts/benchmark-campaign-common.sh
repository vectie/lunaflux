#!/bin/sh

benchmark_campaign_fail() {
  printf '%s\n' "LunaFlux benchmark campaign rejected: $1" >&2
  exit 1
}

benchmark_campaign_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    benchmark_campaign_fail 'no SHA-256 utility is available'
  fi
}

benchmark_campaign_is_sha256() {
  [ "${#1}" -eq 64 ] || return 1
  case "$1" in
    *[!0-9a-f]*|0000000000000000000000000000000000000000000000000000000000000000)
      return 1
      ;;
  esac
}

benchmark_campaign_links() {
  if stat -f '%l' "$1" >/dev/null 2>&1; then
    stat -f '%l' "$1"
  else
    stat -c '%h' "$1"
  fi
}

benchmark_campaign_mode() {
  if stat -f '%Lp' "$1" >/dev/null 2>&1; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

benchmark_campaign_require_file() {
  bcc_path=$1
  bcc_label=$2
  case "$bcc_path" in
    /*) ;;
    *) benchmark_campaign_fail "$bcc_label path is not absolute" ;;
  esac
  case "$bcc_path" in
    *//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
      benchmark_campaign_fail "$bcc_label path is not canonical"
      ;;
  esac
  [ -f "$bcc_path" ] && [ ! -L "$bcc_path" ] ||
    benchmark_campaign_fail "$bcc_label is not a regular non-symlink file"
  [ "$(benchmark_campaign_links "$bcc_path")" = 1 ] ||
    benchmark_campaign_fail "$bcc_label has a hard-link alias"
  bcc_parent=$(CDPATH= cd -- "$(dirname -- "$bcc_path")" && pwd -P)
  [ "$bcc_parent/$(basename -- "$bcc_path")" = "$bcc_path" ] ||
    benchmark_campaign_fail "$bcc_label path contains a directory alias"
}

benchmark_campaign_require_digest() {
  bcc_path=$1
  bcc_digest=$2
  bcc_label=$3
  benchmark_campaign_is_sha256 "$bcc_digest" ||
    benchmark_campaign_fail "$bcc_label digest is invalid"
  [ "$(benchmark_campaign_sha256 "$bcc_path")" = "$bcc_digest" ] ||
    benchmark_campaign_fail "$bcc_label digest does not match its bytes"
}

benchmark_campaign_require_size() {
  bcc_path=$1
  bcc_maximum=$2
  bcc_label=$3
  bcc_bytes=$(wc -c < "$bcc_path" | tr -d ' ')
  [ "$bcc_bytes" -gt 0 ] && [ "$bcc_bytes" -le "$bcc_maximum" ] ||
    benchmark_campaign_fail "$bcc_label size is outside its fixed bound"
}

benchmark_campaign_field() {
  bcc_file=$1
  bcc_line=$2
  bcc_key=$3
  bcc_value=$(sed -n "${bcc_line}p" "$bcc_file")
  case "$bcc_value" in
    "$bcc_key="*) printf '%s\n' "${bcc_value#*=}" ;;
    *) benchmark_campaign_fail "noncanonical field $bcc_key" ;;
  esac
}

benchmark_campaign_require_newline() {
  bcc_file=$1
  bcc_label=$2
  [ -s "$bcc_file" ] || benchmark_campaign_fail "$bcc_label is empty"
  bcc_last=$(tail -c 1 "$bcc_file" | od -An -tu1 | tr -d ' \n')
  [ "$bcc_last" = 10 ] ||
    benchmark_campaign_fail "$bcc_label is not newline terminated"
}

benchmark_campaign_name() {
  case "$1" in
    0) printf '%s\n' latency ;;
    1) printf '%s\n' chat ;;
    2) printf '%s\n' long_prefill ;;
    3) printf '%s\n' decode_heavy ;;
    4) printf '%s\n' prefix_rich ;;
    5) printf '%s\n' prefix_cold ;;
    6) printf '%s\n' saturation ;;
    7) printf '%s\n' churn ;;
    8) printf '%s\n' mixed ;;
    *) benchmark_campaign_fail 'invalid profile index' ;;
  esac
}

benchmark_engine_name() {
  case "$1" in
    0) printf '%s\n' lunaflux ;;
    1) printf '%s\n' vllm ;;
    2) printf '%s\n' sglang ;;
    *) benchmark_campaign_fail 'invalid engine index' ;;
  esac
}

benchmark_engine_for_position() {
  bcc_ordinal=$1
  bcc_position=$2
  case "$bcc_ordinal:$bcc_position" in
    1:0|2:2|3:1) printf '%s\n' 0 ;;
    1:1|2:0|3:2) printf '%s\n' 1 ;;
    1:2|2:1|3:0) printf '%s\n' 2 ;;
    *) benchmark_campaign_fail 'invalid counterbalance coordinate' ;;
  esac
}

benchmark_capture_name() {
  bcc_index=$1
  if [ "$bcc_index" -lt 10 ]; then
    printf 'trial-00%s.capture.json\n' "$bcc_index"
  elif [ "$bcc_index" -lt 100 ]; then
    printf 'trial-0%s.capture.json\n' "$bcc_index"
  else
    benchmark_campaign_fail 'capture index is outside the fixed matrix'
  fi
}

benchmark_campaign_write_inventory() {
  bcc_root=$1
  bcc_paths=$2
  bcc_output=$3
  : > "$bcc_output"
  while IFS= read -r bcc_relative; do
    printf '%s  %s\n' \
      "$(benchmark_campaign_sha256 "$bcc_root/$bcc_relative")" \
      "$bcc_relative" >> "$bcc_output"
  done < "$bcc_paths"
}

benchmark_campaign_validate_inventory() {
  bcc_root=$1
  bcc_inventory=$2
  bcc_expected=$3
  benchmark_campaign_require_newline "$bcc_inventory" 'campaign file inventory'
  bcc_declared=${BENCHMARK_CAMPAIGN_SCRATCH:?}/declared.$$
  : > "$bcc_declared"
  while IFS= read -r bcc_line; do
    bcc_digest=${bcc_line%%  *}
    bcc_relative=${bcc_line#*  }
    [ "$bcc_relative" != "$bcc_line" ] ||
      benchmark_campaign_fail 'campaign inventory line is malformed'
    benchmark_campaign_is_sha256 "$bcc_digest" ||
      benchmark_campaign_fail 'campaign inventory digest is invalid'
    case "$bcc_relative" in
      ''|/*|*/|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
        benchmark_campaign_fail 'campaign inventory path is unsafe'
        ;;
    esac
    [ "$bcc_line" = "$bcc_digest  $bcc_relative" ] ||
      benchmark_campaign_fail 'campaign inventory line is not canonical'
    benchmark_campaign_require_digest "$bcc_root/$bcc_relative" "$bcc_digest" \
      "inventoried campaign file $bcc_relative"
    printf '%s\n' "$bcc_relative" >> "$bcc_declared"
  done < "$bcc_inventory"
  cmp -s "$bcc_declared" "$bcc_expected" ||
    benchmark_campaign_fail 'campaign inventory is not the exact expected path set'
  rm -f "$bcc_declared"
}

# Writes one canonical invocation record from the exact argv that a trusted
# process supervisor will execute. Argument bytes are represented only by
# SHA-256, so separators, whitespace, and empty arguments cannot alias.
benchmark_campaign_write_invocation() {
  bci_output=$1
  bci_label=$2
  bci_timeout=$3
  bci_grace=$4
  bci_credential_scope=$5
  shift 5
  case "$bci_label" in
    ''|*[!A-Za-z0-9._-]*) benchmark_campaign_fail 'invocation label is invalid' ;;
  esac
  case "$bci_credential_scope" in
    none|fd-3|fd-4|fd-5) ;;
    *) benchmark_campaign_fail 'invocation credential scope is invalid' ;;
  esac
  bci_scratch=${BENCHMARK_CAMPAIGN_INVOCATION_SCRATCH:?}
  bci_argument=$bci_scratch/invocation-argument.$$
  {
    printf '%s\n' \
      'schema=lunaflux.external-process-invocation.v1' \
      "label=$bci_label" \
      "timeout_seconds=$bci_timeout" \
      "grace_seconds=$bci_grace" \
      "credential_scope=$bci_credential_scope" \
      "argument_count=$#"
    bci_index=0
    for bci_value in "$@"; do
      printf '%s' "$bci_value" > "$bci_argument"
      printf 'argument_%03d_sha256=%s\n' "$bci_index" \
        "$(benchmark_campaign_sha256 "$bci_argument")"
      bci_index=$((bci_index + 1))
    done
  } > "$bci_output"
  rm -f "$bci_argument"
}

benchmark_campaign_validate_invocation() {
  bcv_file=$1
  bcv_digest=$2
  bcv_label=$3
  bcv_timeout=$4
  bcv_grace=$5
  bcv_scope=$6
  benchmark_campaign_require_file "$bcv_file" "$bcv_label invocation"
  benchmark_campaign_require_digest "$bcv_file" "$bcv_digest" \
    "$bcv_label invocation"
  benchmark_campaign_require_newline "$bcv_file" "$bcv_label invocation"
  [ "$(benchmark_campaign_field "$bcv_file" 1 schema)" = \
      lunaflux.external-process-invocation.v1 ] &&
    [ "$(benchmark_campaign_field "$bcv_file" 2 label)" = "$bcv_label" ] &&
    [ "$(benchmark_campaign_field "$bcv_file" 3 timeout_seconds)" = \
      "$bcv_timeout" ] &&
    [ "$(benchmark_campaign_field "$bcv_file" 4 grace_seconds)" = \
      "$bcv_grace" ] &&
    [ "$(benchmark_campaign_field "$bcv_file" 5 credential_scope)" = \
      "$bcv_scope" ] ||
    benchmark_campaign_fail "$bcv_label invocation authority changed"
  bcv_count=$(benchmark_campaign_field "$bcv_file" 6 argument_count)
  case "$bcv_count" in
    0|[1-9]|[1-9][0-9]) ;;
    *) benchmark_campaign_fail "$bcv_label invocation argument count is invalid" ;;
  esac
  [ "$(wc -l < "$bcv_file" | tr -d ' ')" -eq $((bcv_count + 6)) ] ||
    benchmark_campaign_fail "$bcv_label invocation has the wrong shape"
  bcv_index=0
  while [ "$bcv_index" -lt "$bcv_count" ]; do
    bcv_key=$(printf 'argument_%03d_sha256' "$bcv_index")
    bcv_value=$(benchmark_campaign_field "$bcv_file" $((bcv_index + 7)) \
      "$bcv_key")
    benchmark_campaign_is_sha256 "$bcv_value" ||
      benchmark_campaign_fail "$bcv_label invocation argument digest is invalid"
    bcv_index=$((bcv_index + 1))
  done
}

benchmark_campaign_validate_supervisor_receipt() {
  bcr_receipt=$1
  bcr_invocation_sha=$2
  bcr_label=$3
  benchmark_campaign_require_file "$bcr_receipt" "$bcr_label supervisor receipt"
  benchmark_campaign_require_newline "$bcr_receipt" "$bcr_label supervisor receipt"
  [ "$(wc -l < "$bcr_receipt" | tr -d ' ')" -eq 9 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 1 schema)" = \
      lunaflux.external-process-supervisor.v2 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 2 invocation_sha256)" = \
      "$bcr_invocation_sha" ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 3 outcome)" = completed ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 4 exit_status)" = 0 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 5 timed_out)" = 0 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 6 cancelled)" = 0 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 7 process_group_empty)" = 1 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 8 stdout_closed)" = 1 ] &&
    [ "$(benchmark_campaign_field "$bcr_receipt" 9 stderr_closed)" = 1 ] ||
    benchmark_campaign_fail "$bcr_label supervisor did not prove bounded cleanup"
}
