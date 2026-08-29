#!/bin/sh
set -eu
[ "${1:-}" = --version ] || exit 41
printf '%s\n' 'ptxas release 13.1, V13.1.115'
