#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

cli_source=cmd/lunaflux/native_run.mbt
dispatch_source=cmd/lunaflux/main.mbt
legacy_source=cmd/lunaflux/operator_commands.mbt
test_source=cmd/lunaflux/main_wbtest.mbt
runtime_test=ops/runtime_instance/release_preflight_test.mbt
hostile_output=$(mktemp "${TMPDIR:-/tmp}/lunaflux-operator-cli-hostile.XXXXXX")
trap 'rm -f "$hostile_output"' EXIT HUP INT TERM

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

for anchor in \
  'fn exact_release_diagnostic_argument(' \
  '[_, "doctor", deployment] => Some(ReleaseDoctor(deployment))' \
  '[_, "plan", deployment] => Some(ReleasePlan(deployment))' \
  'Some(ReleaseInspectKernels(deployment))' \
  'Some(command) => run_release_diagnostic(command)' \
  'let admitted = @runtime_instance.preflight_release(' \
  'println(admitted.evidence())' \
  'println("filesystem authority retained: false")' \
  'println("device opened: false")' \
  'println("process spawned: false")' \
  'println("listener bound: false")' \
  'println("readiness: false")'
do
  if ! rg -F -q "$anchor" "$cli_source" "$dispatch_source"; then
    fail "canonical operator diagnostic invariant is missing: $anchor"
  fi
done

for anchor in \
  '"legacy-doctor" => Some("doctor")' \
  '"legacy-plan" => Some("plan")' \
  '"legacy-inspect-kernels" => Some("inspect-kernels")' \
  'LunaFlux legacy compatibility' \
  'only visibly named commands enter legacy compatibility' \
  'release preflight rejects every unpinned or malformed deployment operand'
do
  if ! rg -F -q "$anchor" "$legacy_source" "$test_source" "$runtime_test"; then
    fail "legacy isolation or hostile coverage is missing: $anchor"
  fi
done

# Bare diagnostic operands belong exclusively to the authenticated grammar.
# No exact bare command may dispatch to model_startup or host probing.
if rg -n \
  '\[_, "(doctor|plan|inspect-kernels)"(, [^]]+)?\] => (print_doctor|print_doctor_json|run_operator_command)' \
  "$dispatch_source"; then
  fail 'canonical diagnostics can fall through to legacy preflight'
fi

for command in doctor plan inspect-kernels
do
  if moon run --target native cmd/lunaflux -- "$command" MODEL \
    >"$hostile_output" 2>&1; then
    fail "unpinned canonical $command operand was accepted"
  fi
  if rg -q \
    '(available: true|readiness: true|legacy compatibility)' \
    "$hostile_output"; then
    fail "unpinned canonical $command operand produced a success-like claim"
  fi
done

moon info --target native ops/runtime_instance cmd/lunaflux
moon check \
  --target native --deny-warn --warn-list +73 \
  ops/runtime_instance cmd/lunaflux
moon test \
  --target native --deny-warn --warn-list +73 \
  cmd/lunaflux/main_wbtest.mbt \
  ops/runtime_instance/release_preflight_test.mbt

printf '%s\n' 'LunaFlux canonical operator CLI convergence is valid.'
