# Rank-group wire

`engine/rank_group_wire` is the neutral framed control protocol between one
same-host tensor-parallel parent supervisor and one rank child. It owns no
process, pipe, socket, scheduler plan, device, collective, filesystem, module,
or transport authority.

The v1 binding repeats the exact model identity, nonzero model-plan and group
generations, rank/world coordinates, and the canonical group, topology,
worker, rank-device-plan, collective, and rank-KV SHA-256 digests already
published by tensor-parallel bootstrap admission. Launch-contract and artifact
bundle owners do not currently publish canonical bundle digests; v1 therefore
does not manufacture weaker digests from their public views. A future schema
must version forward when those owning packages publish exact digests.

Every frame contains the full binding. Configure, Submit, and completion
payloads are bounded opaque bytes: this package checks framing, length,
checksum, and transcript identity, but deliberately does not decode bootstrap,
scheduler, or worker-wire types. Configure must carry a nonempty local-
admission payload; the eventual child execution owner must still decode and
authenticate it before `Ready` at its owning bootstrap boundary.

Every frame carries an explicit parent-to-child or child-to-parent direction.
The transcript is a thread-confined, authority-free group ordering validator.
It requires the parent to configure every rank before it accepts any child
`Ready`; a child may emit `Ready` only after its local execution manifest,
launch contracts, artifact bundle, and rank bootstrap have all been admitted.
It then accepts one semantic plan at a time in either preallocated physical
slot A or B. A submitted plan may be polled repeatedly
through `PollComplete -> Pending`, completed through
`PollComplete -> Complete`, or failed through one of the payload-free
`PollComplete -> PlanFailed(CollectiveFailure)` and
`PollComplete -> PlanFailed(RankExecutionFailure)` responses. Wire codes 13
and 14 are distinct so the failure category is authenticated by the frame
rather than carried in mutable diagnostics. Either failure retires the exact
active sequence into `AbortedPlan`; it cannot be substituted with `Complete`
and may only proceed through drain and close. A supervisor-directed
`Abort -> Aborted` remains the cooperative termination path when every child
can answer. Clean closure is
`Drain -> Drained -> Close -> Closed`. Any stale generation, binding, slot,
sequence, payload, direction, or ordering substitution sticky-faults the
transcript. Abort is legal from every configured nonterminal phase, including
while a completion read is pending. A physical A/B slot is reusable only after
every rank has acknowledged exact completion or abort for that generation,
sequence, and slot.

Transport EOF or child-process loss is outside the wire transcript's ACK
protocol. The supervisor must immediately call `abandon_on_transport_loss`,
delegate the exact lost-rank event to `engine/rank_group_protocol`, and never
wait for an ACK from the unavailable child. The wire transcript only latches a
payload-free terminal `TransportLost` rule; it does not duplicate rank-loss,
drain-exclusion, replacement, or cleanup ownership.

Buffers are allocated only during construction. Encoding, loading, transcript
acceptance, checksum validation, and payload copying reuse caller- or
startup-owned fixed storage. Stateless operations encode directly into caller
storage and copy authenticated header scalars. First-Configure decoding copies
the binding into startup-owned storage and may allocate there; steady-state
validation against that locked binding allocates nothing. No stateless decoded
value retains its source storage. Validated FrameBuffer views authenticate
their exact buffer epoch and become stale after the next encode/load; those
methods delegate the same codec core.

`RankGroupWireRankTranscript` is the corresponding authority-free child-rank
validator. Its first Configure copies and locks one exact binding; local
bootstrap admission must then match that binding and supply the authenticated
predecessor before Ready. Later parent commands and child responses use
direction-specific acceptors, so a reflected response cannot advance command
state. Its sequencing and rank-zero/follower completion rules are
differential-tested against the group transcript.

The wire remains one canonical v1 schema: this Phase-7 protocol has not
shipped, so adding the two exact failure kinds does not retain a compatibility
branch. A surviving rank that did not report a peer failure may receive
`Drain` while its local transcript is still `Executing` or `Polling`; that
command retires the same active sequence exactly once and does not fabricate
an `Aborted` acknowledgement for the failed peer.
