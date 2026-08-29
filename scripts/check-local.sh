#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

usage() {
  printf '%s\n' \
    'usage: scripts/check-local.sh PACKAGE_OR_MOONBIT_FILE [...]' \
    '' \
    'Runs only the fast local MoonBit loop:' \
    '  moon fmt --check TARGETS...' \
    '  moon check --target native --deny-warn --warn-list +73 TARGETS...' \
    '  moon test --target native --deny-warn --warn-list +73 TARGETS...' \
    '' \
    'It does not run physical CUDA, sanitizers, soaks, evidence/release' \
    'assembly, benchmark campaigns, or deployment/network approval gates.'
}

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

if [ "$1" = --help ] || [ "$1" = -h ]; then
  usage
  exit 0
fi

printf 'local check: format (%s target(s))\n' "$#"
moon fmt --check "$@"
printf 'local check: native warnings (%s target(s))\n' "$#"
moon check --target native --deny-warn --warn-list +73 "$@"
printf 'local check: focused tests (%s target(s))\n' "$#"
moon test --target native --deny-warn --warn-list +73 "$@"
printf '%s\n' 'local check passed'
