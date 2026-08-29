#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
package_dir="$repo_root/kernels/luna_profile_priority"

actual_imports=$(awk '
  /^import \{/ { if (seen == 0) { inside=1; seen=1; next } }
  inside && /^}/ { exit }
  inside { print }
' "$package_dir/moon.pkg" |
  sed -E 's/^[[:space:]]*"([^"]+)".*/\1/' |
  sort)
expected_imports=$(printf '%s\n' \
  moonbitlang/core/encoding/utf8 \
  moonbitlang/x/crypto \
  vectie/lunaflux/engine/device_profile \
  vectie/lunaflux/kv/device_layout \
  vectie/lunaflux/kernels/catalog \
  vectie/lunaflux/kernels/launch_contract \
  vectie/lunaflux/model/plan \
  vectie/lunaflux/model/spec |
  sort)
if [ "$actual_imports" != "$expected_imports" ]; then
  printf '%s\n' \
    'profile priority admission crosses into runtime, service, or native authority' >&2
  printf 'actual production imports:\n%s\n' "$actual_imports" >&2
  exit 1
fi

if rg -n \
  'cupti|nvml|nvrtc|cuModuleLoadDataEx|popen[[:space:]]*\(|system[[:space:]]*\(' \
  "$package_dir" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' \
    'profile priority admission must not capture profiles or compile kernels at runtime' >&2
  exit 1
fi

for required in \
  'lunaflux.kernel-profile-priority.v1' \
  'declared_attributed_self_time_ns' \
  'ObservationOrder' \
  'plan.kv_execution() != StatelessFullContext' \
  'LunaKernelProfileUnsupported(ExecutionMode)' \
  'luna_profile_checked_add' \
  'ReadOnlyArray::from_array(observations)' \
  'operation.required_capability()'
do
  if ! rg -F -q "$required" "$package_dir" --glob '*.mbt'; then
    printf '%s\n' "profile priority invariant is missing: ${required}" >&2
    exit 1
  fi
done

for required in \
  'lunaflux.kernel-paged-profile-priority.v1' \
  'pub fn admit_luna_paged_kernel_profile(' \
  'plan.kv_execution() != PagedKeyValue' \
  'launches.model_identity() != plan.identity()' \
  'launches.device_kv_layout()' \
  'launches.device_target()' \
  'launches.scope()' \
  'launches.profiles()' \
  'luna_paged_launch_covers_operation' \
  'shape.page_table_block_count() > profile.max_page_table_entries()' \
  'row.context_length() > maximum_context' \
  'ReadOnlyArray::from_array(rows.to_owned())' \
  'ReadOnlyArray::from_array(observations.to_owned())' \
  'LunaPagedPageTableTraceDigest' \
  'page_table_trace_digest'
do
  if ! rg -F -q "$required" "$package_dir" --glob '*.mbt'; then
    printf '%s\n' "paged profile priority invariant is missing: ${required}" >&2
    exit 1
  fi
done

if ! rg -F -q \
  'The capture is prioritization evidence only.' \
  "$package_dir/README.mbt.md"; then
  printf '%s\n' 'profile priority documentation overstates its authority' >&2
  exit 1
fi

if rg -n \
  '^pub fn [A-Za-z0-9_:]*(execute|launch|load|compile|ready|dispatch)\(|^pub (struct|enum) [A-Za-z0-9_]*Ready' \
  "$package_dir" --glob '*.mbt'; then
  printf '%s\n' \
    'profile priority public surface gained execution or readiness authority' >&2
  exit 1
fi

for source in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  if [ "$lines" -gt 500 ]; then
    printf '%s\n' \
      "profile priority source exceeds 500 lines: $source ($lines)" >&2
    exit 1
  fi
done

cd "$repo_root"
moon check --target native --deny-warn kernels/luna_profile_priority
moon test --target native --deny-warn kernels/luna_profile_priority
printf '%s\n' 'Luna profile-priority boundary is valid.'
