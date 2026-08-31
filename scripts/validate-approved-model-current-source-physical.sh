#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
runner=scripts/run-approved-model-current-source-physical.sh
archiver=scripts/create-portable-source-archive.sh
process_group=scripts/physical-campaign-process-group.sh
evidence_helper=scripts/immutable-evidence-directory.sh
failed=0

fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

for script in "$runner" "$archiver" "$process_group" "$evidence_helper"; do
  [ -f "$script" ] || fail "current-source campaign script is missing: $script"
  bash -n "$script" || fail "current-source campaign script is invalid: $script"
done

for anchor in \
  'COPYFILE_DISABLE=1' \
  'tar --no-xattrs' \
  "-name '._*'" \
  'special source entry exists:' \
  "grep -Eq '(^|/)\\._[^/]*$'" \
  "--exclude=\"\$repo_name/.git\"" \
  "--exclude=\"\$repo_name/_build\"" \
  "--exclude=\"\$repo_name/trace.json\""; do
  grep -Fq -- "$anchor" "$archiver" ||
    fail "portable source guard is missing: $anchor"
done

for anchor in \
  'evidence directory must be outside the source tree' \
  'special source entry exists:' \
  'source.files.sha256' \
  'source.symlinks.v1' \
  'source files changed while building the physical release' \
  'source files changed while materializing the physical campaign' \
  'setsid is required for deterministic campaign cleanup' \
  'lunaflux_stop_campaign_group' \
  "trap '' HUP INT TERM" \
  'evidence_sealed=0' \
  'surviving_process_group=%s' \
  'moon build --target native --release --deny-warn cmd/device_worker_child' \
  '_build/native/release/build/cmd/device_worker_child/device_worker_child.exe' \
  'scripts/build-luna-bf16-kernel-set.sh' \
  'CUDA toolchain identity inspection failed' \
  'scripts/verify-luna-bf16-kernel-set.sh' \
  'scripts/validate-approved-model-context-churn.sh' \
  'scripts/validate-approved-model-long-context.sh' \
  'scripts/validate-online-listener-restart-boundary.sh' \
  'scripts/materialize-approved-tiny-bf16-launch.sh' \
  '"$worker#sha256=$worker_sha"' \
  'materialized child bytes do not match the current-source child' \
  'prefix-reuse-v1' \
  'prefix launch child bytes do not match the current-source child' \
  'host-sampling-referee-v1' \
  'spawned-device-greedy-readback' \
  '"$campaign" device-greedy' \
  'sampling_result_row_bytes=8' \
  'selected_logit_correctness=pass' \
  'lunaflux_run_tracked_campaign "$logs/spawned.stdout"' \
  'lunaflux_run_tracked_campaign "$logs/serving.stdout"' \
  'lunaflux_run_tracked_campaign \' \
  '"$logs/openai-qualification.stdout"' \
  '"$campaign" openai-qualification' \
  'lunaflux_run_tracked_campaign "$logs/benchmark.stdout"' \
  'lunaflux_run_tracked_campaign "$logs/prefix.stdout"' \
  '"$campaign" prefix-reuse' \
  'lunaflux_run_tracked_campaign \' \
  '"$logs/context-churn.stdout" "$logs/context-churn.stderr"' \
  '"$campaign" context-churn' \
  '"$logs/long-context.stdout" "$logs/long-context.stderr"' \
  '"$campaign" long-context' \
  'event_order=accepted,token,token,usage,completed' \
  'bearer_auth_missing=401' \
  'post_drain_admission=refused' \
  'second_cached_input_tokens=8' \
  'campaign_scope=qualification-only' \
  'release_promotion=not-claimed' \
  'broader_context_length_coverage=not-established' \
  'actual_input_token_lengths=8,9' \
  'beyond_9_token_input=not-established' \
  'full_256_token_context=not-established' \
  'input_tokens_physically_processed=51' \
  'verify_canonical_evidence \' \
  'context_churn_qualification_sha256 36' \
  'actual_long_context_qualification_sha256 47' \
  'gpu-inventory.before' \
  'GPU inventory or framebuffer use changed across campaign' \
  'request_path_jit=0' \
  'lunaflux_prepare_evidence_manifest "$evidence_dir"' \
  'lunaflux_seal_evidence_directory'; do
  grep -Fq -- "$anchor" "$runner" ||
    fail "current-source physical invariant is missing: $anchor"
done

for anchor in \
  'lunaflux_prepare_evidence_manifest()' \
  'lunaflux_seal_evidence_directory()' \
  'find "$evidence_dir" -type f -exec chmod 0444 {} +' \
  'find "$evidence_dir" -type d -exec chmod 0555 {} +'; do
  grep -Fq -- "$anchor" "$evidence_helper" ||
    fail "immutable evidence helper invariant is missing: $anchor"
done

if rg -ni 'nvrtc|--ptx|output=ptx|runtime.{0,12}compil' "$runner"; then
  fail 'current-source spawned campaign introduced PTX or runtime compilation'
fi
if rg -n 'WORKER(_PATH|_SHA)?=|expected_worker_sha256=|\$\{?3\}?' "$runner"; then
  fail 'current-source campaign accepts caller-supplied worker authority'
fi

runner_build=$(rg -n 'moon build --target native --release --deny-warn cmd/device_worker_child' \
  "$runner" | cut -d: -f1)
runner_digest=$(rg -n '^worker_sha=\$\(sha256_file "\$worker"\)' "$runner" |
  cut -d: -f1)
runner_materialize=$(rg -n '^scripts/materialize-approved-tiny-bf16-launch.sh' \
  "$runner" | cut -d: -f1 | sed -n '1p')
runner_spawn=$(rg -n '^  "\$campaign" "\$launch_root#sha256=\$launch_sha"' \
  "$runner" | cut -d: -f1)
runner_serve=$(rg -n '^  "\$campaign" serving ' "$runner" | cut -d: -f1)
runner_openai=$(rg -n '^  "\$campaign" openai-qualification ' "$runner" |
  cut -d: -f1)
runner_prefix=$(rg -n '^  "\$campaign" prefix-reuse ' "$runner" | cut -d: -f1)
runner_context_churn=$(rg -n '^  "\$campaign" context-churn ' "$runner" |
  cut -d: -f1)
runner_long_context=$(rg -n '^  "\$campaign" long-context ' "$runner" |
  cut -d: -f1)
runner_final_balance=$(rg -n '^stage=final-balance$' "$runner" | cut -d: -f1)
runner_context_validator=$(rg -n '^scripts/validate-approved-model-context-churn.sh ' \
  "$runner" | cut -d: -f1)
runner_long_validator=$(rg -n '^scripts/validate-approved-model-long-context.sh ' \
  "$runner" | cut -d: -f1)
runner_toolchain=$(rg -n '^stage=toolchain-identity$' "$runner" | cut -d: -f1)
if [ -z "$runner_build" ] || [ -z "$runner_digest" ] ||
  [ -z "$runner_materialize" ] || [ -z "$runner_spawn" ] ||
  [ -z "$runner_serve" ] || [ -z "$runner_openai" ] ||
  [ -z "$runner_prefix" ] || [ -z "$runner_context_churn" ] ||
  [ -z "$runner_long_context" ] || [ -z "$runner_final_balance" ] ||
  [ -z "$runner_context_validator" ] || [ -z "$runner_long_validator" ] ||
  [ -z "$runner_toolchain" ] ||
  [ "$runner_context_validator" -ge "$runner_toolchain" ] ||
  [ "$runner_long_validator" -ge "$runner_toolchain" ] ||
  [ "$runner_build" -ge "$runner_digest" ] ||
  [ "$runner_digest" -ge "$runner_materialize" ] ||
  [ "$runner_materialize" -ge "$runner_spawn" ] ||
  [ "$runner_spawn" -ge "$runner_serve" ] ||
  [ "$runner_serve" -ge "$runner_openai" ] ||
  [ "$runner_openai" -ge "$runner_prefix" ] ||
  [ "$runner_prefix" -ge "$runner_context_churn" ] ||
  [ "$runner_context_churn" -ge "$runner_long_context" ] ||
  [ "$runner_long_context" -ge "$runner_final_balance" ]; then
  fail 'current-source child build/digest/materialize/execute order is invalid'
fi

for mode in context-churn; do
  mode_line=$(rg -n '^  "\$campaign" '"$mode"' ' "$runner" | cut -d: -f1)
  log_line=$(sed -n "$((mode_line - 1))p" "$runner")
  tracked_line=$(sed -n "$((mode_line - 2))p" "$runner")
  authority_line=$(sed -n "$((mode_line + 1))p" "$runner")
  [ "$tracked_line" = 'lunaflux_run_tracked_campaign \' ] ||
    fail "current-source $mode bypasses tracked process-group ownership"
  [ "$log_line" = \
    "  \"\$logs/$mode.stdout\" \"\$logs/$mode.stderr\" \\" ] ||
    fail "current-source $mode lost its dedicated immutable evidence streams"
  [ "$authority_line" = \
    '  "$launch_root#sha256=$launch_sha" "$worker_sha" ||' ] ||
    fail "current-source $mode lost exact launch/child authority"
done

long_context_log_line=$(sed -n "$((runner_long_context - 1))p" "$runner")
long_context_tracked_line=$(sed -n "$((runner_long_context - 2))p" "$runner")
long_context_authority_line=$(sed -n "$((runner_long_context + 1))p" "$runner")
[ "$long_context_tracked_line" = 'lunaflux_run_tracked_campaign \' ] ||
  fail 'current-source long-context bypasses tracked process-group ownership'
[ "$long_context_log_line" = \
  '  "$logs/long-context.stdout" "$logs/long-context.stderr" \' ] ||
  fail 'current-source long-context lost its dedicated immutable evidence streams'
[ "$long_context_authority_line" = \
  '  "$prefix_launch_root#sha256=$prefix_launch_sha" "$worker_sha" ||' ] ||
  fail 'current-source long-context lost its exact max-input-9 launch authority'

for mode in context-churn long-context; do
  mode_count=$(rg -c '^  "\$campaign" '"$mode"' ' "$runner")
  [ "$mode_count" -eq 1 ] ||
    fail "current-source campaign must execute $mode exactly once"
done
if rg -n \
  'release_promotion=pass|broader_context_length_coverage=pass|beyond_9_token_input=pass|128_or_255_token_execution=pass|full_256_token_context=pass|gpu_allocator_instrumentation=pass' \
  "$runner"; then
  fail 'current-source context qualification broadened into a promotion claim'
fi

scratch=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-current-source-gate.XXXXXX")
cleanup() { rm -rf -- "$scratch"; }
trap cleanup EXIT HUP INT TERM
. "$process_group"
if command -v setsid >/dev/null 2>&1; then
  campaign_status=0
  lunaflux_run_tracked_campaign "$scratch/orphan.stdout" "$scratch/orphan.stderr" \
    sh -c 'trap "" TERM; (trap "" TERM; sleep 30) & exit 17' || campaign_status=$?
  [ "$campaign_status" -eq 17 ] ||
    fail 'tracked campaign did not preserve the failed leader status'
  if kill -0 -- "-$lunaflux_last_campaign_pgid" 2>/dev/null; then
    fail 'tracked campaign left a descendant process group alive'
  fi
fi
if "$runner" >/dev/null 2>&1; then
  fail 'current-source campaign accepted missing arguments'
fi
if "$runner" relative-nvcc "$scratch/new-evidence" >/dev/null 2>&1; then
  fail 'current-source campaign accepted relative NVCC authority'
fi
mkdir "$scratch/existing"
if "$runner" /usr/bin/true "$scratch/existing" >/dev/null 2>&1; then
  fail 'current-source campaign accepted an existing evidence directory'
fi
if "$runner" /usr/bin/true "$repo_root/inside-source" >/dev/null 2>&1; then
  fail 'current-source campaign accepted evidence inside the source tree'
fi
if "$archiver" relative.tar.gz >/dev/null 2>&1; then
  fail 'portable archiver accepted a relative output path'
fi

scripts/validate-approved-model-spawned-physical.sh >/dev/null ||
  fail 'spawned physical child-substitution boundary failed'
scripts/validate-approved-model-context-churn.sh >/dev/null ||
  fail 'context-churn qualification ownership boundary failed'
scripts/validate-approved-model-long-context.sh >/dev/null ||
  fail 'actual long-context qualification ownership boundary failed'
scripts/validate-approved-model-serving-readiness.sh >/dev/null ||
  fail 'native listener boundary failed'
scripts/validate-openai-qualification-ownership.sh >/dev/null ||
  fail 'OpenAI qualification ownership boundary failed'

if [ "$failed" -ne 0 ]; then
  exit 1
fi
printf '%s\n' 'Approved-model current-source physical campaign boundary passed.'
