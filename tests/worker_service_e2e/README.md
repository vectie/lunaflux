# Worker-service end-to-end gate

This executable joins the real scheduler, worker service, process supervisor,
and separately linked deterministic worker child. It proves two outstanding
plans, output-publication backpressure with exact frame retry, orderly child
close, scheduler worker-failure retirement, non-reusing replacement startup,
and a successful post-restart completion with balanced KV ownership.

The child remains a deterministic protocol implementation rather than a CUDA
worker. Device-backed readiness is a separate physical release gate.

The persistent-online slice uses a startup-sized `LunaRequestPreparationPool`:
external submission yields opaque work, one central owner advances bounded
steps, and Ready work publishes a destructively transferable prepared claim.
Its evidence covers single-lane storage-generation reuse, Saturated/Draining
receipt non-consumption, online Busy/Draining prepared-shell non-consumption,
lower-admission rollback, healthy retirement, recovery close, and stale
Work/Prepared/Claim rejection after reuse. Sequential requests in one child,
foreign preparation binding, queued expiry, stale tickets, and post-admission
failure reuse remain covered. The event evidence additionally
proves single-issued opaque credits, delayed stale-alias rejection after the
next event is live, Usage-to-terminal epoch separation, abort invalidation, and
framed-adapter composition. The harness releases the transport frame before it
ACKs the semantic credit; the adapter never receives ACK authority.

After warm pool startup, proportional tokenization/copy/output-setup progress
is allocation-free. Ready assembly may allocate only constant-size
`TokenizedRequest` and claim shells, and `take_prepared` may allocate one
constant prepared shell; the generated-C gate filters exactly those owners and
rejects payload arrays, byte/text construction, or growing collections.

The reusable native pipeline evidence writes two complete canonical frames in
one socket operation. It proves both requests are live before output, the real
worker receives one two-row/two-token plan, and the globally ordered stream
publishes both Accepted events before the oldest request is cancelled. The
cancelled request's Usage/Completed pair precedes the surviving request's
Token/Usage/Completed sequence. After every byte is confirmed, the same
connection reaches an observable balanced boundary; peer close retires only
that Stream, and listener-first drain ends with zero queue, active requests,
and used KV pages.

The disjoint `--balance-10000 WORKER` long-run balance gate keeps one rooted
worker service alive for exactly 10,000
monotonically identified requests with fixed active/waiting slots, natural
completion, preflight cancellation, queued expiry, terminal-ring backpressure,
ten periodic empty-worker recoveries across 1,000 waves, and final cooperative
shutdown/reap. Its release run reconciles every ID plus empty physical/pending
work, publication rings, block tables, and KV ownership.

The opt-in `--soak-24h WORKER` gate drives the frozen native pipeline server
for at least 86,400,000 monotonic milliseconds. It submits deterministic
two-request coalesced waves at approximately one request per second through
1-to-17-byte client writes, a 4,096-byte bounded read envelope, and a 17-byte
server write quantum. It delays
all event reads until the authenticated per-wave balanced boundary, periodically
attempts to cancel the oldest request, retires and reconnects the peer, and injects a
malformed framed request whose rejection must close only that connection.
Every wave reconciles its two monotonically derived request IDs and exact
ordered event shapes, then proves queue depth, active requests, and used KV
pages are zero while the initial free-page count is restored. Final
listener-first drain requires equal accept/disconnect counters, no pending
request capability, and a cooperatively closed/reaped worker child. The v2
policy records cumulative two-live, two-row-batched, output-backpressured,
scheduled-cancel, attempted-cancel, accepted-cancel, and actually cancelled waves
independently. A two-live observation triggers one public cancellation attempt;
an exact coordinator rejection is a non-mutating missed cancellation, so that
wave completes naturally. An accepted request may also lose to an already
natural terminal; terminal metrics, not the call return, own the actual outcome.
Each functional observation, including at least one actual cancellation, must
occur in the full run, but a balanced wave may
serialize into one-row plans; batching and cancellation rates are separate
performance benchmarks, not per-wave correctness assertions. The frozen v1
run aborted on that former assertion and is not promotion evidence; a fresh
complete v2 24-hour run is required.

The promotion runner accepts no duration argument or environment override. It
must be invoked explicitly as:

```sh
LUNAFLUX_RUN_PHASE23_SOAK_24H=1 scripts/run-phase23-soak-24h.sh
```

`--soak-smoke WORKER ABSOLUTE_FIXTURE_ROOT` is a separate bounded development
check of the same wave,
cancel, malformed-rejection, reconnect, delayed-output, accounting, and final
drain machinery. A smoke pass is not 24-hour promotion evidence.

`--soak-cycle-fast WORKER ABSOLUTE_FIXTURE_ROOT` and
`--soak-cycle-timer WORKER ABSOLUTE_FIXTURE_ROOT` are
separate, cycle-bounded failure-localization tools. Both replay 2,300 waves
from cycle zero with the production wave and periodic predicate helpers while
printing scalar-only phase and balance breadcrumbs. The fast form removes wall
pacing; the timer form registers exactly one one-millisecond async sleep after
each completed cycle. Neither form is byte-equivalent to a frozen promotion
binary, and neither result is promotion evidence.
