#!/bin/sh
set -eu

package="engine/rank_child_control"
api="$package/pkg.generated.mbti"

actual_imports=$(sed -n 's/.*"vectie\/lunaflux\/\([^"]*\)".*/\1/p' \
  "$package/moon.pkg" | sort)
expected_imports=$(printf '%s\n' \
  'engine/rank_group_wire' \
  'engine/worker_wire' \
  'internal/process' \
  'model/spec' | sort)
if [ "$actual_imports" != "$expected_imports" ]; then
  echo "rank-child control dependency boundary changed" >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/(scheduler|device|internal/(cuda|nccl)|service|runtime|kernels)' \
  "$package/moon.pkg" "$package"/*.mbt; then
  echo "rank-child control crossed scheduler, backend, service, or artifact ownership" >&2
  exit 1
fi

if rg -n '\b(Array|Map|HashMap)::|:\s*Array\[|:\s*Map\[|ArrayView\[' \
  "$package"/*.mbt | rg -v '_test|_wbtest'; then
  echo "rank-child control production path uses dynamic or borrowed storage" >&2
  exit 1
fi

if rg -n ':\s*\([^)]*\)\s*->|\(\)\s*->' "$package"/*.mbt | \
  rg -v '_test|_wbtest'; then
  echo "rank-child control production path introduced callbacks" >&2
  exit 1
fi

if rg -n 'ApprovedRoot|Device(Context|Allocation|Stream)|Communicator|NCCL|Nccl|Cuda|Scheduler|SubmittedSchedulePlan|ValidatedPlanFrame|Thread|spawn' \
  "$package" --glob '*.mbt'; then
  echo "rank-child control leaks root, backend, scheduler, or thread authority" >&2
  exit 1
fi

for required in \
  'rx_destination: FixedArray::make(frame_bytes' \
  'rx_staging: FixedArray::make(frame_bytes' \
  'tx_storage: FixedArray::make(frame_bytes' \
  'begin_read_frame_or_eof(' \
  'progress_read_frame_or_eof()' \
  'begin_write_frame(' \
  'progress_write_frame()' \
  'self.receive_held = true' \
  'if self.send_active {' \
  'inbound_epoch != self.1' \
  'self.transcript.admit_ready(' \
  'self.transcript.accept_parent(' \
  'self.transcript.accept_child(' \
  'pub fn RankChildControl::begin_collective_failed(' \
  'PlanFailed(CollectiveFailure)' \
  'pub fn RankChildControl::begin_execution_failed(' \
  'PlanFailed(RankExecutionFailure)' \
  'pub fn RankChildControl::begin_graph_telemetry_sidecar(' \
  'encode_worker_graph_telemetry(' \
  'report.sequence_value() != self.transcript.previous_sequence_value()' \
  'self.transcript.abandon_on_transport_loss()'; do
  if ! rg -Fq "$required" "$package"; then
    echo "rank-child control invariant is missing: $required" >&2
    exit 1
  fi
done

if rg -n 'pub (struct|enum).*(@process|InheritedChannel|Process)|pub fn.*(@process|InheritedChannel|Process)' \
  "$package"/*.mbt; then
  echo "rank-child control public source leaks process authority" >&2
  exit 1
fi

if rg -n 'pub fn RankChildControl::begin_complete\(' "$package"/*.mbt; then
  echo "rank-child control collapsed leader/follower completion authority" >&2
  exit 1
fi

if [ -f "$api" ]; then
  if rg -n '@process|InheritedChannel|Process(Error|Limits)|rx_destination|rx_staging|tx_storage|channel :' "$api"; then
    echo "rank-child control generated API leaks storage or process authority" >&2
    exit 1
  fi
  for required in \
    'pub fn open_rank_child_control(' \
    'pub fn RankChildControl::begin_receive(' \
    'pub fn RankChildControl::progress_receive(' \
    'pub fn RankChildControl::begin_ready(' \
    'pub fn RankChildControl::accept_submit(' \
    'pub fn RankChildControl::begin_pending(' \
    'pub fn RankChildControl::begin_leader_complete(' \
    'pub fn RankChildControl::begin_follower_complete(' \
    'pub fn RankChildControl::begin_graph_telemetry_sidecar(' \
    'pub fn RankChildControl::begin_aborted(' \
    'pub fn RankChildControl::begin_drained(' \
    'pub fn RankChildControl::begin_closed(' \
    'pub fn RankChildControl::progress_send('; do
    if ! rg -Fq "$required" "$api"; then
      echo "rank-child control generated API lost focused surface: $required" >&2
      exit 1
    fi
  done
  for opaque_type in RankChildControl RankChildReceiveProgress \
    RankChildSendProgress; do
    if ! awk -v type="$opaque_type" '
      $0 == "pub struct " type " {" { seen = 1; next }
      seen && $0 == "  // private fields" { private_fields = 1; next }
      seen && $0 == "}" { exit !(private_fields == 1) }
    ' "$api"; then
      printf 'rank-child control public type is not opaque: %s\n' \
        "$opaque_type" >&2
      exit 1
    fi
  done
  if rg -n 'RankChildInbound::new|impl Debug for RankChild(Control|Inbound)' "$api"; then
    echo "rank-child control API exposes capability fabrication/debugging" >&2
    exit 1
  fi
  if rg -n 'pub fn RankChildControl::begin_complete\(' "$api"; then
    echo "rank-child generated API collapsed completion roles" >&2
    exit 1
  fi
fi

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line rank-child budget" >&2
    exit 1
  fi
done

echo "rank-child control boundaries: ok"
