#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

if rg -n 'extern\s+"[cC]"|#external|native-stub' runtime/inherited_drain; then
  printf '%s\n' 'public inherited-drain wrapper must own no native ABI' >&2
  exit 1
fi

if [ "$(rg -c 'extern\s+"[cC]"' internal/inherited_drain/ffi.mbt)" -ne 7 ] ||
  rg -n 'extern\s+"[cC]"|#external' internal/inherited_drain \
    --glob '*.mbt' --glob '!ffi.mbt'; then
  printf '%s\n' 'inherited-drain native declarations drifted from one exact FFI file' >&2
  exit 1
fi

for required in \
  'LF_DRAIN_FIXED_FD 5' \
  'AF_UNIX' \
  'SOCK_STREAM' \
  'F_DUPFD_CLOEXEC' \
  'O_NONBLOCK' \
  'MSG_PEEK' \
  'lunaflux_inherited_drain_close'; do
  if ! rg -q "$required" internal/inherited_drain/inherited_drain.c; then
    printf 'inherited-drain ABI lost required anchor: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'bind\s*\(|listen\s*\(|accept\s*\(|connect\s*\(|getenv\s*\(|system\s*\(|popen\s*\(' \
  internal/inherited_drain runtime/inherited_drain; then
  printf '%s\n' 'inherited drain must not bind, connect, discover ambient state, or run a shell' >&2
  exit 1
fi

if ! rg -q 'vectie/lunaflux/internal/inherited_drain' \
    runtime/inherited_drain/moon.pkg ||
  rg -n 'internal/inherited_drain' --glob 'moon.pkg' \
    --glob '!runtime/inherited_drain/moon.pkg' \
    --glob '!internal/inherited_drain/moon.pkg'; then
  printf '%s\n' 'only the public inherited-drain wrapper may import its ABI owner' >&2
  exit 1
fi

moon info >/dev/null
if rg -n 'internal/inherited_drain|InheritedDrainHandle|NativeInheritedDrain' \
  runtime/inherited_drain/pkg.generated.mbti \
  ops/runtime_instance/pkg.generated.mbti; then
  printf '%s\n' 'internal inherited-drain concrete authority leaked publicly' >&2
  exit 1
fi

for source_file in internal/inherited_drain/*.mbt \
  internal/inherited_drain/*.c runtime/inherited_drain/*.mbt; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; inherited-drain files must stay below 500\n' \
      "$source_file" "$line_count" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux inherited-drain ABI boundary is valid.'
