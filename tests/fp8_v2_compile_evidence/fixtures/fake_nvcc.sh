#!/bin/sh
set -eu

mode_file=$(dirname -- "$0")/mode
mode=quiet
[ ! -f "$mode_file" ] || mode=$(sed -n '1p' "$mode_file")
if [ "${1:-}" = --version ]; then
  if [ "$mode" = wrong-version ]; then
    printf '%s\n' 'Cuda compilation tools, release 13.1, V13.1.114'
  else
    printf '%s\n' 'Cuda compilation tools, release 13.1, V13.1.115'
  fi
  exit 0
fi

output=
target=
previous=
for argument in "$@"; do
  if [ "$previous" = -o ]; then output=$argument; fi
  case "$argument" in -arch=*) target=${argument#-arch=} ;; esac
  previous=$argument
done
[ -n "$output" ] && {
  [ "$target" = sm_89 ] || [ "$target" = sm_90 ] || [ "$target" = sm_120 ]
} || exit 41
case "$mode" in
  quiet) printf '\177ELF\002\001\001\000synthetic-%s\n' "$target" >"$output" ;;
  noisy)
    printf '\177ELF\002\001\001\000synthetic-%s\n' "$target" >"$output"
    printf '%s\n' 'synthetic compiler noise'
    ;;
  nonelf) printf 'not-elf-%s\n' "$target" >"$output" ;;
  fail) exit 42 ;;
  *) exit 43 ;;
esac
