#!/bin/sh
set -eu

if [ "${1:-}" = --version ]; then
  printf '%s\n' 'nvdisasm release 13.1, V13.1.115 fake qualification fixture'
  exit 0
fi

[ "$#" = 1 ] || exit 2
printf '%s\n' \
  '/*0000*/ MOV R0, R0;' \
  '/*0010*/ LDG.E R2, [R4];' \
  '/*0020*/ STG.E [R4], R2;'
