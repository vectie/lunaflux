#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if rg -n 'extern\s+"[cC]"|#external|native-stub' runtime/inference_credential; then
  printf '%s\n' 'public inference-credential wrapper must own no native ABI' >&2
  exit 1
fi

if [ "$(rg -c 'extern\s+"[cC]"' internal/inference_credential/ffi.mbt)" -ne 6 ] ||
  rg -n 'extern\s+"[cC]"|#external' internal/inference_credential \
    --glob '*.mbt' --glob '!ffi.mbt'; then
  printf '%s\n' 'inference-credential declarations drifted from one exact FFI file' >&2
  exit 1
fi

for required in \
  'LF_CREDENTIAL_FIXED_FD 6' \
  'AF_UNIX' \
  'SOCK_STREAM' \
  'F_DUPFD_CLOEXEC' \
  'O_NONBLOCK' \
  'MSG_PEEK' \
  'lunaflux_inference_credential_close'; do
  if ! rg -q "$required" internal/inference_credential/inference_credential.c; then
    printf 'inference-credential ABI lost required anchor: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'bind\s*\(|listen\s*\(|accept\s*\(|connect\s*\(|getenv\s*\(|system\s*\(|popen\s*\(' \
  internal/inference_credential runtime/inference_credential; then
  printf '%s\n' 'inference credential must not bind, discover ambient state, or execute' >&2
  exit 1
fi

if ! rg -q 'vectie/lunaflux/internal/inference_credential' \
    runtime/inference_credential/moon.pkg ||
  rg -n 'internal/inference_credential' --glob 'moon.pkg' \
    --glob '!runtime/inference_credential/moon.pkg' \
    --glob '!internal/inference_credential/moon.pkg' \
    --glob '!**/_build/**'; then
  printf '%s\n' 'only the public credential wrapper may import its ABI owner' >&2
  exit 1
fi

moon info >/dev/null
if rg -n 'internal/inference_credential|InferenceCredentialHandle|NativeInferenceCredential' \
  runtime/inference_credential/pkg.generated.mbti \
  ops/runtime_instance/pkg.generated.mbti; then
  printf '%s\n' 'internal credential authority leaked publicly' >&2
  exit 1
fi

for source_file in internal/inference_credential/*.mbt \
  internal/inference_credential/*.c runtime/inference_credential/*.mbt; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; credential files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux inherited inference-credential ABI boundary is valid.'
