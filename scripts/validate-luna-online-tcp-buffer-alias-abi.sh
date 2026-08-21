#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

if ! rg -F -x -q 'supported_targets = "native"' \
    internal/online_tcp_buffer_alias/moon.pkg ||
  ! rg -F -x -q '  "native-stub": [ "alias.c" ],' \
    internal/online_tcp_buffer_alias/moon.pkg ||
  ! rg -F -x -q \
    '  "stub-cc-flags": "-std=c11 -Wall -Wextra -Werror",' \
    internal/online_tcp_buffer_alias/moon.pkg; then
  printf '%s\n' \
    'online TCP buffer-alias ABI must remain native-only with one strict stub' >&2
  failed=1
fi

stub_count=$(rg -o '"[A-Za-z0-9_]+\.c"' \
  internal/online_tcp_buffer_alias/moon.pkg | wc -l | tr -d ' ')
extern_count=$(rg -n 'extern\s+"[cC]"' \
  internal/online_tcp_buffer_alias --glob '*.mbt' | wc -l | tr -d ' ')
if [ "$stub_count" -ne 1 ] || [ "$extern_count" -ne 1 ] ||
  ! rg -q --pcre2 -U \
    '#borrow\(source\)\nextern "c" fn raw_retain_bytes_as_fixed_array\(\n  source : Bytes,\n\) -> FixedArray\[Byte\] = "lunaflux_online_tcp_retain_bytes_as_fixed_array"' \
    internal/online_tcp_buffer_alias/alias.mbt; then
  printf '%s\n' \
    'online TCP alias bridge ownership annotation or exact ABI surface drifted' >&2
  failed=1
fi

alias_body=$(sed -n \
  '/^moonbit_bytes_t lunaflux_online_tcp_retain_bytes_as_fixed_array(/,/^}/p' \
  internal/online_tcp_buffer_alias/alias.c)
if [ -z "$alias_body" ] ||
  [ "$(printf '%s\n' "$alias_body" | rg -c 'moonbit_incref\(source\)')" -ne 1 ] ||
  [ "$(printf '%s\n' "$alias_body" | rg -c '^  return source;$')" -ne 1 ] ||
  printf '%s\n' "$alias_body" | rg -q \
    'moonbit_decref|moonbit_make|malloc|calloc|realloc|free|memcpy|memmove|blit'; then
  printf '%s\n' \
    'online TCP alias bridge must retain once and return the identical pointer' >&2
  failed=1
fi

expected_interface='pub fn retain_bytes_as_fixed_array(Bytes) -> FixedArray[Byte]'
actual_interface=$(rg '^pub ' \
  internal/online_tcp_buffer_alias/pkg.generated.mbti || true)
if [ "$actual_interface" != "$expected_interface" ] ||
  rg -n '^pub (struct|enum|type|trait)' \
    internal/online_tcp_buffer_alias/pkg.generated.mbti; then
  printf '%s\n' 'online TCP alias ABI interface drifted or exposed a type' >&2
  failed=1
fi

importers=$(rg -l '"vectie/lunaflux/internal/online_tcp_buffer_alias"' \
  --glob 'moon.pkg' 2>/dev/null || true)
if [ "$importers" != 'service/online_tcp/moon.pkg' ]; then
  printf '%s\n%s\n' \
    'online TCP alias ABI has an unauthorized production importer:' \
    "$importers" >&2
  failed=1
fi

callers=$(rg -l '@buffer_alias\.retain_bytes_as_fixed_array\(' \
  --glob '*.mbt' 2>/dev/null || true)
if [ "$callers" != 'service/online_tcp/scratch.mbt' ] ||
  [ "$(rg -c '@buffer_alias\.retain_bytes_as_fixed_array\(immutable\)' \
    service/online_tcp/scratch.mbt)" -ne 1 ] ||
  ! rg -q --pcre2 -U \
    "let immutable = Bytes::makei\\(capacity, _ => b'\\\\x00'\\)\\n  let mutable = @buffer_alias\\.retain_bytes_as_fixed_array\\(immutable\\)" \
    service/online_tcp/scratch.mbt; then
  printf '%s\n' \
    'online TCP alias ABI must have one dynamic-Bytes service caller' >&2
  failed=1
fi

if rg -n 'BytesView|Bytes::from|b"|@async|Thread|spawn' \
    internal/online_tcp_buffer_alias \
    service/online_tcp/scratch.mbt service/online_tcp/scratch_types.mbt \
    --glob '*.mbt'; then
  printf '%s\n' \
    'online TCP alias owner acquired a static/view caller or cross-thread path' >&2
  failed=1
fi

for source_file in \
  internal/online_tcp_buffer_alias/alias.c \
  internal/online_tcp_buffer_alias/alias.mbt \
  internal/online_tcp_buffer_alias/alias_probe.c; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s\n' "$source_file" \
      'online TCP alias boundary file exceeds 500 lines' >&2
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux online TCP buffer-alias ABI boundary is valid.'
