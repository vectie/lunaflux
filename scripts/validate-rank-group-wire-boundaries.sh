#!/bin/sh
set -eu

package="engine/rank_group_wire"
api="$package/pkg.generated.mbti"

if rg -n 'vectie/lunaflux/(scheduler|device|internal|service|runtime|deploy|engine/worker|engine/rank_group_protocol|kernels)' \
  "$package/moon.pkg" "$package"/*.mbt; then
  echo "rank-group wire crossed a scheduler, backend, owner, or artifact boundary" >&2
  exit 1
fi

if rg -n 'ChildProcess|Socket|Tcp|ApprovedRoot|Device(Context|Allocation|Stream)|Communicator|Nccl|NCCL|Scheduler|SubmittedSchedulePlan|ValidatedPlanFrame' \
  "$package" --glob '*.mbt'; then
  echo "rank-group wire leaks transport, scheduler, or backend authority" >&2
  exit 1
fi

if rg -n '\b(Array|Map|HashMap)::|:\s*Array\[|:\s*Map\[' \
  "$package"/*.mbt | rg -v 'fixture_test|_test|_wbtest'; then
  echo "rank-group wire production path uses dynamic collection storage" >&2
  exit 1
fi

if rg -n 'for [^\n]* in \[' "$package"/*.mbt | \
  rg -v 'fixture_test|_test|_wbtest'; then
  echo "rank-group wire production path contains an allocating collection literal" >&2
  exit 1
fi

for required in \
  'const RANK_GROUP_WIRE_HEADER_BYTES : Int = 644' \
  'const RANK_GROUP_WIRE_VERSION : UInt = 2' \
  'const RANK_GROUP_WIRE_MAX_PAYLOAD_BYTES : Int = 67108220' \
  'priv execution_plan_digest : String' \
  'pub fn RankGroupWireBinding::execution_plan_digest(' \
  'execution_plan_digest=decode_wire_digest_string(source, 448)' \
  'collective_digest=decode_wire_digest_string(source, 512)' \
  'kv_digest=decode_wire_digest_string(source, 576)' \
  'wire_write_u32(destination, 640, 0U)' \
  'pub(all) enum RankGroupWireDirection' \
  'if direction != expected_direction(kind)' \
  'self.phase_count != self.world_size' \
  'fn RankGroupWireTranscript::begin_abort' \
  'self.lifecycle != Executing && self.lifecycle != Polling' \
  'pub fn RankGroupWireTranscript::load_and_accept' \
  'self.poison(error_rule(error))' \
  'pub fn RankGroupWireTranscript::abandon_on_transport_loss' \
  'self.poison(TransportLost)' \
  'pub fn decode_rank_group_wire_configure(' \
  'pub fn validate_rank_group_wire_binding(' \
  'pub fn encode_rank_group_wire_control_into(' \
  'pub fn encode_rank_group_wire_payload_into(' \
  'pub fn RankGroupWireRankTranscript::accept_parent(' \
  'pub fn RankGroupWireRankTranscript::accept_child(' \
  'PlanFailed(CollectiveFailure) => 13' \
  'PlanFailed(RankExecutionFailure) => 14' \
  'pub fn RankGroupWireTranscript::plan_failure(' \
  'pub fn RankGroupWireTranscript::plan_failure_rank(' \
  'pub fn RankGroupWireRankTranscript::plan_failure(' \
  'self.previous_sequence = self.active_sequence' \
  'self.lifecycle_value == Aborting' \
  'self.0.epoch != self.1'; do
  if ! rg -Fq "$required" "$package"; then
    echo "rank-group wire invariant is missing: $required" >&2
    exit 1
  fi
done


if rg -n 'lost_rank|lost_ranks|mark_rank_lost|report_rank_loss' \
  "$package" --glob '*.mbt'; then
  echo "rank-group wire duplicated lost-rank ownership" >&2
  exit 1
fi

if [ "${1:-}" != "--static-only" ] && [ -f "$api" ]; then
  if rg -n 'ChildProcess|Socket|Tcp|ApprovedRoot|Device(Context|Allocation|Stream)|Communicator|Scheduler|SubmittedSchedulePlan' "$api"; then
    echo "rank-group wire generated API leaks owner authority" >&2
    exit 1
  fi
  for required in \
    'pub fn RankGroupWireBinding::new(' \
    'pub fn RankGroupWireBinding::execution_plan_digest(' \
    'pub fn RankGroupWireFrameBuffer::encode_control(' \
    'pub fn RankGroupWireFrameBuffer::encode_payload(' \
    'pub fn RankGroupWireFrameBuffer::encode_configure_payload(' \
    'pub fn decode_rank_group_wire_configure(' \
    'pub fn validate_rank_group_wire_binding(' \
    'pub fn encode_rank_group_wire_control_into(' \
    'pub fn encode_rank_group_wire_payload_into(' \
    'pub fn RankGroupWireTranscript::accept(' \
    'pub fn RankGroupWireRankTranscript::accept_parent(' \
    'pub fn RankGroupWireRankTranscript::accept_child(' \
    'pub fn RankGroupWireTranscript::abandon_on_transport_loss(' \
    'pub fn RankGroupWireTranscript::load_and_accept('; do
    if ! rg -Fq "$required" "$api"; then
      echo "rank-group wire generated API lost its focused surface: $required" >&2
      exit 1
    fi
  done
  for opaque_type in RankGroupWireBinding RankGroupWireFrameBuffer \
    RankGroupWireLimits RankGroupWireTranscript \
    RankGroupWireDecodedHeader RankGroupWireDecodedConfigure \
    RankGroupWireRankTranscript; do
    if ! awk -v type="$opaque_type" '
      $0 == "pub struct " type " {" { seen = 1; next }
      seen && $0 == "  // private fields" { private_fields = 1; next }
      seen && $0 == "}" { exit !(private_fields == 1) }
    ' "$api"; then
      printf 'rank-group wire public type is not opaque: %s\n' \
        "$opaque_type" >&2
      exit 1
    fi
  done
  if rg -n 'ValidatedRankGroupWireFrame::new|RankGroupWireDecoded(Header|Configure)::new|impl Debug for RankGroupWire(Binding|FrameBuffer|Limits|Transcript)' \
    "$api"; then
    echo "rank-group wire API exposes frame fabrication or owner debugging" >&2
    exit 1
  fi
fi

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line wire package budget" >&2
    exit 1
  fi
done

if [ "${1:-}" = "--static-only" ]; then
  echo "rank-group wire static boundaries: ok"
  exit 0
fi

echo "rank-group wire boundaries: ok"
