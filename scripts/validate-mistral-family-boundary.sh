#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
builder="$root/model/mistral"
reader="$root/model/config_reader"
scheduler="$root/scheduler"

if rg -n 'vectie/lunaflux/(scheduler|kv|prefix|api|service|device|kernels|internal/cuda)' \
  "$builder/moon.pkg" >/dev/null; then
  echo "Mistral builder crosses a forbidden runtime boundary" >&2
  exit 1
fi

if rg -n '(Mistral|mistral|sliding_window)' "$scheduler" --glob '*.mbt' \
  --glob 'moon.pkg' >/dev/null; then
  echo "scheduler contains a Mistral-family branch" >&2
  exit 1
fi

if ! rg -n 'parse_mistral_(bytes|string)' "$reader/reader.mbt" >/dev/null; then
  echo "strict Mistral configuration admission is missing" >&2
  exit 1
fi

if ! rg -n '@llama\.build(_paged)?\(' "$builder/builder.mbt" >/dev/null; then
  echo "Mistral does not reuse the authenticated dense operation graph" >&2
  exit 1
fi

if rg -n '(new_with_kv_execution|PlanOperation::new|OperationKind)' \
  "$builder/builder.mbt" >/dev/null; then
  echo "Mistral builder introduced a parallel operation graph" >&2
  exit 1
fi

echo "Mistral family boundary: ok"
