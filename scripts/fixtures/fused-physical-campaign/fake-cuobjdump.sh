#!/bin/sh
set -eu

if [ "${1:-}" = --version ]; then
  printf '%s\n' 'cuobjdump release 13.1, V13.1.115 fake qualification fixture'
  exit 0
fi

case "${1:-}" in
  --dump-sass)
    printf '%s\n' \
      '/*0000*/ MOV R0, R0;' \
      '/*0010*/ LDG.E R2, [R4];' \
      '/*0020*/ STG.E [R4], R2;'
    ;;
  --dump-resource-usage)
    case "${2:-}" in
      *residual*) shared=512 ;;
      *) shared=0 ;;
    esac
    printf 'REG:64 SHARED:%s STACK:0 LOCAL:0\n' "$shared"
    ;;
  *) exit 2 ;;
esac
