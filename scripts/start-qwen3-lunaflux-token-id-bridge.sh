#!/bin/sh
set -eu

if [ "$#" -ne 12 ]; then
  printf '%s\n' \
    'usage: start-qwen3-lunaflux-token-id-bridge.sh BRIDGE_EXECUTABLE#sha256=HEX TOKENIZER_JSON#sha256=HEX LAUNCH#sha256=HEX C32_CAPACITY_RECEIPT#sha256=HEX LISTEN_ADDR RUNTIME_ADDR MODEL_CONTENT_SHA256 MODEL_PLAN_SHA256 MAX_INPUT_TOKENS MAX_OUTPUT_TOKENS MAX_TOKEN_ID MAX_CONTEXT_TOKENS' >&2
  exit 2
fi

bridge_identity=$1
case $bridge_identity in
  /*'#sha256='????????????????????????????????????????????????????????????????) ;;
  *) printf '%s\n' 'bridge executable identity is malformed' >&2; exit 2 ;;
esac
bridge_executable=${bridge_identity%#sha256=*}
[ -f "$bridge_executable" ] && [ ! -L "$bridge_executable" ] &&
  [ -x "$bridge_executable" ] || {
    printf '%s\n' 'bridge executable is not a regular executable' >&2
    exit 2
  }

export LC_ALL=C
exec "$bridge_executable" "$@"
