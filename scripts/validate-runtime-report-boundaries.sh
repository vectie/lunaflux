#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

report_dir=ops/runtime_report
failed=0

fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

if [ ! -f "$report_dir/moon.pkg" ] ||
  [ ! -f "$report_dir/build.mbt" ] ||
  [ ! -f "$report_dir/render.mbt" ]; then
  fail 'runtime report package is incomplete'
fi

if rg -n \
  'vectie/lunaflux/(device"|internal/|runtime/approved_fs)' \
  "$report_dir/moon.pkg"; then
  fail 'runtime report gained filesystem, internal, or device authority'
fi

public_api=$(rg -n '^pub(\(all\))? (struct Runtime(Instance)?Report|suberror RuntimeReportError|fn (build|build_instance|render|render_instance)\()' \
  "$report_dir" --glob '*.mbt' || true)
public_api_count=$(printf '%s\n' "$public_api" | sed '/^$/d' | wc -l | tr -d ' ')
if [ "$public_api_count" -ne 7 ]; then
  printf '%s\n%s\n' 'runtime report public API changed:' "$public_api" >&2
  failed=1
fi

all_public=$(rg -n '^pub(\(all\))? (struct|enum|suberror|fn)' \
  "$report_dir" --glob '*.mbt' || true)
all_public_count=$(printf '%s\n' "$all_public" | sed '/^$/d' | wc -l | tr -d ' ')
if [ "$all_public_count" -ne 7 ]; then
  printf '%s\n%s\n' 'runtime report exposes an unreviewed public symbol:' \
    "$all_public" >&2
  failed=1
fi

if [ ! -f "$report_dir/pkg.generated.mbti" ] ||
  ! rg -U -q \
    'pub struct RuntimeReport \{\n  // private fields\n\}' \
    "$report_dir/pkg.generated.mbti" ||
  ! rg -U -q \
    'pub struct RuntimeInstanceReport \{\n  // private fields\n\}' \
    "$report_dir/pkg.generated.mbti"; then
  fail 'runtime report generated interface is not opaque'
fi

if ! rg -q 'checked_nonnegative_add' "$report_dir/build.mbt" ||
  ! rg -q 'weight_reserved_bytes' "$report_dir/build.mbt" ||
  ! rg -q 'activation_workspace_reserved_bytes' "$report_dir/build.mbt" ||
  ! rg -q 'kv_reserved_bytes' "$report_dir/build.mbt"; then
  fail 'runtime report checked reserved-total formula disappeared'
fi

if ! rg -q 'kernel_module_file_bytes_excluded_from_device_total' \
  "$report_dir/render.mbt" ||
  ! rg -q 'scheduler.capacity: unavailable' "$report_dir/render.mbt" ||
  ! rg -q 'service.capacity: unavailable' "$report_dir/render.mbt" ||
  ! rg -q 'service.openai_activation: not admitted' "$report_dir/render.mbt"; then
  fail 'runtime report exclusion or unavailable-capacity labels disappeared'
fi

if rg -n '(arena_bytes|activation_bytes|total_module_bytes)\(' \
  cmd/lunaflux --glob '*.mbt'; then
  fail 'CLI performs ad-hoc runtime memory arithmetic'
fi

if ! rg -q 'let report = @runtime_report\.build\(admission\) catch' \
  cmd/lunaflux/operator_commands.mbt ||
  ! rg -q '@runtime_report\.render\(report\)' \
    cmd/lunaflux/operator_commands.mbt; then
  fail 'pinned operator commands no longer consume the typed runtime report'
fi

if ! rg -q '"legacy-doctor" => Some\("doctor"\)' \
  cmd/lunaflux/operator_commands.mbt ||
  ! rg -q 'legacy compatibility physical diagnostic' \
    cmd/lunaflux/operator_commands.mbt ||
  ! rg -q '"doctor" => "diagnostic report available: true"' \
    cmd/lunaflux/operator_commands.mbt; then
  fail 'pinned doctor physical-diagnostic/report form disappeared'
fi

if ! rg -q 'admission: Some\(admission\)' \
  ops/model_startup/pinned_preflight.mbt ||
  ! rg -q 'println\("readiness: false"\)' \
    cmd/lunaflux/operator_commands.mbt; then
  fail 'pinned diagnostic no longer preserves host evidence with false readiness'
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'runtime report boundary gate passed'
