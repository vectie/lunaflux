#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

source_file=cmd/lunaflux/native_run.mbt
dispatch_file=cmd/lunaflux/main.mbt

line_of() {
  pattern=$1
  file=$2
  rg -n -m 1 "$pattern" "$file" | cut -d: -f1
}

prepare_line=$(line_of 'let drain = @inherited_drain\.prepare_inherited_drain_v1\(\)' "$source_file")
owner_line=$(line_of 'let owner = @runtime_instance\.prepare_opaque_cli\(' "$source_file")
attach_line=$(line_of 'owner\.attach_inherited_drain_v1\(drain\)' "$source_file")
serve_line=$(line_of 'let mut ready_printed = false' "$source_file")

if [ -z "$prepare_line" ] || [ -z "$owner_line" ] || [ -z "$attach_line" ] ||
  [ -z "$serve_line" ] || [ "$prepare_line" -ge "$owner_line" ] ||
  [ "$owner_line" -ge "$attach_line" ] || [ "$attach_line" -ge "$serve_line" ]; then
  printf '%s\n' \
    'opaque CLI must acquire inherited drain before opaque runtime preparation and attach it before progress' >&2
  exit 1
fi

grammar_count=$(rg -c '^fn exact_opaque_run_argument\(' "$source_file" || true)
grammar_branch_count=$(rg -c '^[[:space:]]*\[_, "run", model\] => Some\(model\)$' \
  "$source_file" || true)
dispatch_match_count=$(rg -c \
  '^[[:space:]]*match exact_opaque_run_argument\(arguments\) \{$' \
  "$dispatch_file" || true)
dispatch_branch_count=$(rg -c \
  '^[[:space:]]*Some\(model\) => run_native_model\(model\)$' \
  "$dispatch_file" || true)
if [ "$grammar_count" -ne 1 ] || [ "$grammar_branch_count" -ne 1 ] ||
  [ "$dispatch_match_count" -ne 1 ] || [ "$dispatch_branch_count" -ne 1 ]; then
  printf '%s\n' 'one-argument run no longer selects the inherited-drain runtime path' >&2
  exit 1
fi

# Near-name helpers or alternate direct one-argument branches must not bypass
# the exact grammar owner above.
if rg -n 'exact_opaque_run_argument_[A-Za-z0-9_]*\(' \
  "$source_file" "$dispatch_file" ||
  rg -n '^[[:space:]]*\[_, "run", [A-Za-z_][A-Za-z0-9_]*\] =>' \
  "$dispatch_file"; then
  printf '%s\n' 'one-argument run acquired an alternate dispatch surface' >&2
  exit 1
fi

if rg -n 'getenv|@env|listen|bind|connect|http|token|secret' \
  runtime/inherited_drain ops/runtime_instance/inherited_drain_owner.mbt; then
  printf '%s\n' 'inherited drain acquired ambient, network-listener, HTTP, or secret authority' >&2
  exit 1
fi

moon build cmd/lunaflux --target native --release --deny-warn

generated_c=_build/native/release/build/cmd/lunaflux/lunaflux.c
if [ ! -f "$generated_c" ]; then
  generated_c=$(find _build/native/release/build/cmd/lunaflux \
    -maxdepth 1 -name '*.c' -print | head -n 1)
fi
if [ -z "$generated_c" ] || [ ! -f "$generated_c" ]; then
  printf '%s\n' 'opaque CLI release C output is missing' >&2
  exit 1
fi

for anchor in \
  'prepare__inherited__drain__v1' \
  'prepare__opaque__cli' \
  'attach__inherited__drain__v1' \
  'run__native__model'; do
  if ! rg -q "$anchor" "$generated_c"; then
    printf 'opaque CLI release output lost inherited-drain anchor: %s\n' \
      "$anchor" >&2
    exit 1
  fi
done

printf '%s\n' 'LunaFlux inherited-drain opaque CLI activation boundary is valid.'
