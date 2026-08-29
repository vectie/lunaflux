#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

for validator in \
  scripts/validate-service-event-wire.sh \
  scripts/validate-service-tokenizer-output.sh \
  scripts/validate-service-request-admission.sh \
  scripts/validate-service-authority-surfaces.sh \
  scripts/validate-service-online-session.sh \
  scripts/validate-service-online-tcp.sh
do
  if ! "$validator"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux service and wire boundaries are valid.'
