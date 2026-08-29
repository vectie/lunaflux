#!/bin/sh

set -eu

if [ "$#" -gt 1 ]; then
  printf '%s\n' 'usage: validate-debt-policy.sh [repository-root]' >&2
  exit 2
fi

if [ "$#" -eq 1 ]; then
  repo_root=$(CDPATH= cd -- "$1" && pwd)
else
  repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fi

failed=0
scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-debt-policy.XXXXXX")
trap 'rm -rf "$scratch_dir"' EXIT HUP INT TERM

# Keep this inventory structural. Policy applies to first-party implementation
# and validation source, not dependency caches, build products, vendored trees,
# or generated MoonBit interfaces. Materialize it once so traversal errors are
# never hidden by a successful downstream sort and every check sees one view.
inventory_unsorted="$scratch_dir/source-inventory.unsorted"
inventory="$scratch_dir/source-inventory"
if ! find "$repo_root" \
  \( -type d \( \
    -name .git -o \
    -name _build -o \
    -name .mooncakes -o \
    -name .repos -o \
    -name node_modules -o \
    -name vendor -o \
    -name third_party \
  \) -prune \) -o \
  \( -type f \( \
    -name '*.mbt' -o \
    -name '*.c' -o \
    -name '*.h' -o \
    -name '*.cc' -o \
    -name '*.cpp' -o \
    -name '*.cu' -o \
    -name '*.cuh' -o \
    -name '*.sh' -o \
    -name moon.pkg -o \
    -name moon.mod \
  \) -print \) > "$inventory_unsorted"; then
  printf '%s\n' 'first-party source inventory traversal failed' >&2
  failed=1
fi
LC_ALL=C sort "$inventory_unsorted" > "$inventory"

source_inventory() {
  sed -n 'p' "$inventory"
}

moonbit_inventory() {
  source_inventory | while IFS= read -r source_file; do
    case "$source_file" in
      *.mbt) printf '%s\n' "$source_file" ;;
    esac
  done
}

# Preserve the stronger completed MoonBit split below 500 lines. All other
# first-party code must remain at or below the policy's hard 800-line ceiling.
# awk NR counts an unterminated final logical line; wc -l does not.
source_inventory | while IFS= read -r source_file; do
  line_count=$(awk 'END { print NR + 0 }' "$source_file")
  relative_file=${source_file#"$repo_root"/}
  case "$source_file" in
    *.mbt)
      if [ "$line_count" -ge 500 ]; then
        printf '%s: %s lines; first-party MoonBit files must remain below 500 lines\n' \
          "$relative_file" "$line_count" >&2
        printf '%s\n' debt_policy_failure
      fi
      ;;
    *)
      if [ "$line_count" -gt 800 ]; then
        printf '%s: %s lines; first-party code must remain at or below 800 lines\n' \
          "$relative_file" "$line_count" >&2
        printf '%s\n' debt_policy_failure
      fi
      ;;
  esac
done > "$scratch_dir/lines"

if [ -s "$scratch_dir/lines" ]; then
  failed=1
fi

# Temporary-work marker spellings are reserved throughout first-party source,
# including literals. This makes the rule independent of language/comment
# syntax. A marker is admissible only when its ownership, removal condition,
# and final permitted phase are recorded at the marker itself.
source_inventory | while IFS= read -r source_file; do
  relative_file=${source_file#"$repo_root"/}
  awk -v file="$relative_file" '
    BEGIN { marker = "TO" "DO" }
    index($0, marker) {
      rest = $0
      owned = marker "\\(owner=[A-Za-z0-9_.@/-]+; remove_when=[^;]+; latest_phase=[A-Za-z0-9_.@/-]+\\)"
      while (match(rest, owned)) {
        rest = substr(rest, 1, RSTART - 1) substr(rest, RSTART + RLENGTH)
      }
      if (index(rest, marker)) {
        printf "%s:%d: unowned temporary-work marker; require owner, removal condition, and latest phase\\n", file, NR > "/dev/stderr"
        found = 1
      }
    }
    END { exit found ? 1 : 0 }
  ' "$source_file" || printf '%s\n' debt_policy_failure
done > "$scratch_dir/todo"

if [ -s "$scratch_dir/todo" ]; then
  failed=1
fi

# Shortcut/fix markers are release-gate failures in token-step/control hot
# paths and every production native-ABI ownership package. Marker spellings are
# assembled so this policy implementation does not exempt itself.
release_marker_pattern='H''ACK|FIX''ME'
source_inventory | while IFS= read -r source_file; do
  relative_file=${source_file#"$repo_root"/}
  case "$relative_file" in
    scheduler/*|kv/*|prefix/*|sampling/*|kernels/*|\
    engine/device_step/*|engine/device_worker/*|engine/worker_service/*|\
    service/request_admission/*|service/online_session/*|\
    internal/approved_fs/*|internal/cuda/*|internal/inference_credential/*|\
    internal/monotonic_clock/*|\
    internal/nccl/*|internal/online_tcp_buffer_alias/*|internal/process/*)
      if LC_ALL=C grep -E -n "$release_marker_pattern" "$source_file" >&2; then
        printf '%s: forbidden release marker in hot-path/native-ABI source\n' \
          "$relative_file" >&2
        printf '%s\n' debt_policy_failure
      fi
      ;;
  esac
done > "$scratch_dir/release-markers"

if [ -s "$scratch_dir/release-markers" ]; then
  failed=1
fi

# Column-zero mutable bindings are MoonBit package globals. Function-local
# mutable bindings are indented and remain valid implementation details.
moonbit_inventory | while IFS= read -r source_file; do
  relative_file=${source_file#"$repo_root"/}
  awk -v file="$relative_file" '
    /^(pub(\([^)]*\))?[[:space:]]+|priv[[:space:]]+)?let[[:space:]]+mut[[:space:]]+/ {
      printf "%s:%d: forbidden global mutable MoonBit runtime declaration\\n", file, NR > "/dev/stderr"
      found = 1
    }
    END { exit found ? 1 : 0 }
  ' "$source_file" || printf '%s\n' debt_policy_failure
done > "$scratch_dir/globals"

if [ -s "$scratch_dir/globals" ]; then
  failed=1
fi

# Reject only exact vague dumping-ground names. Cohesive names such as
# prepare_helpers.mbt remain valid and are judged by the same line budget.
moonbit_inventory | while IFS= read -r source_file; do
  relative_file=${source_file#"$repo_root"/}
  case "/$relative_file" in
    */misc/*|*/util/*|*/utils/*|*/misc.mbt|*/util.mbt|*/utils.mbt|\
    */helper.mbt|*/helpers.mbt)
      printf '%s: vague dumping-ground name; use a cohesive responsibility name\n' \
        "$relative_file" >&2
      printf '%s\n' debt_policy_failure
      ;;
  esac
done > "$scratch_dir/names"

if [ -s "$scratch_dir/names" ]; then
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'LunaFlux technical-debt policy is valid.'
