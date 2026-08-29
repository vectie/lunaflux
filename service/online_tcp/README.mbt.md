# Luna native online TCP transport

`LunaOnlineTcpServer` is the authoritative reusable native-framed transport.
It owns one persistent listener and the same long-lived
`LunaOnlineFramedService` across sequential connections. Exactly one accepted
socket and one exact-generation Stream may be active. An accept timeout is a
nonterminal Listening result: it neither consumes the Service nor opens a
Stream. A socket is installed before the Service opens its Stream, with no
await between those owner mutations.

The existing `bind_luna_online_tcp_server` is the max-one compatibility mode:
once its first frame receipt completes, a coalesced following frame remains a
private tail and is discarded during connection retirement. The additive
`bind_luna_online_tcp_pipeline_server` keeps that same socket and Stream live
and admits bounded coalesced or later frames into the fixed request lanes.
Events remain in the Service's single global order. After a terminal event is
fully confirmed and ACKed, pipeline mode exposes one balanced Connected
transition before beginning another socket read; it never overlaps the retired
request with a rebased transport wait. Pipeline concurrency is bounded by the
prepared Service's fixed request lanes and still uses one serialized socket and
runtime task; it is not concurrent-client serving or long-run soak evidence. A
typed rejection, malformed input,
EOF, or read/write failure remains connection-terminal in both modes. The
listener and healthy worker are then reused, and no invented Luna event is
emitted for a rejection.

Production runtime selection uses the authenticated preparation lane count,
which instance admission has already proved equal to the scheduler's total
request-slot capacity. Capacity one retains the compatible singleton native or
OpenAI server. Any larger admitted capacity selects the corresponding
startup-preallocated connection pool. Each pool owns one service and semantic
stream, one fixed parser/output/receipt cell per connection, and bounded
round-robin progress; a partial or backpressured peer cannot own another
client's slot or route. Readiness, drain, cold-start publication, scheduler
metrics, request observations, and deterministic cleanup remain projections of
that single pool owner.

`progress_on_reactor` is the sole socket and Service progress owner. It performs
one bounded transport/owner transition per call and drives cooperative worker
maintenance on that same async task. There is no off-reactor Server method.
`begin_drain` only sets intent; the next reactor transition closes the listener
before disconnecting a Stream or beginning Service drain. Pending finite
observations survive peer/error disconnect, are recorded, then ACKed before the
Stream can retire and the next connection can open.

The Server owns fixed ingress, framed output, Offer, Flight, Stream, and
observation slots allocated at startup. A positive `write_once` result confirms
the exact Offer before its Flight releases the dual-view scratch. Zero, unknown,
timed-out, cancelled, or failed writes never confirm bytes. All connection tail,
cursor, Offer, Flight, receipt, observation-stage, and backpressure-episode
scalars reset before the next accept.

Network accept metrics count committed socket+Stream pairs, not kernel accepts
rolled back before Stream commit. Network disconnect and backpressure are
recorded once per committed connection/episode. Observation counters are staged
before ACK so an ACK retry cannot duplicate them. Monotonic log timestamps are
clamped to the last recorded value on clock failure/backward motion without
discarding the exact finite reason or blocking cleanup.

Queue depth includes the scheduler waiting count plus framed-Service queued and
Prepared authorities. Active requests and KV used/free pages come from an
authenticated scalar scheduler projection. The same projection carries every
prefix lookup/hit/miss/eviction/reuse/computation/publication counter and the
current prefix entry/page gauges. One startup-owned telemetry bridge is shared
by the native and OpenAI server owners; it applies monotonic counter deltas
exactly once, rejects rollback, preserves a fixed KV total, and rejects plan
sequence/shape drift. Batch row/token histograms are
sampled once for each new committed nonzero plan sequence. Request, first-token,
and inter-token latency histograms consume only generation-authenticated
relative timing pulses from the framed coordinator; framed bytes are never
decoded or guessed for telemetry. A once-only server method accepts the future
operator owner's measured root-open-to-listener-ready cold-start interval and
does not confer readiness. While telemetry is temporarily
unavailable, reactor cleanup continues and live snapshots reject rather than
publish stale gauges. Detected monotonic drift is sticky for the live owner.
Closed snapshots expose the last counters/histograms with
exact zero active/queue/used gauges and the startup total as free pages.

Every public call requires exclusive, serialized handoff on the same MoonBit
runtime task. Mutable activity booleans detect retained aliases within that
contract; they are not atomic locks and do not make simultaneous OS-thread
access safe. The fixed-capacity pools arbitrate multiple sockets cooperatively
on that one task; they are not a listener fleet or an unbounded multithreaded
reactor. HTTP/OpenAI support is a distinct additive owner described below; it
does not alter or decode the native framed path.

## Loopback operational HTTP owner

`LunaOnlineHttpControlServer` is a separate observational listener bound only
to an internally selected `127.0.0.1` ephemeral port. Its admitted address is
published for an embedding owner; this is not a public-routability, TLS, or
network-policy claim. It serves only bodyless `GET /healthz` and `GET /readyz`
with fixed payload-safe responses and never parses inference traffic or exposes
drain, authentication, tenant, scheduler, or configuration routes.

Health is 200 only while both the control owner and its singular inference
source are not sticky-failed or closed. Readiness additionally requires a live
control listener, a live ready production inference listener/service, and no
drain intent. Qualification provenance can therefore expose health but never a
200 readiness response. Source drain intent flips readiness before the source
listener closes; the control listener remains available during source drain
and then closes deterministically after it.

Control accept, read, and write operations have an independent one-millisecond
poll cap even when inference transport limits are longer. An incomplete or
slow control client is retired locally without poisoning listener health, so a
client cannot hold the serialized runtime owner through an inference-scale I/O
wait. All parser storage and fixed responses are allocated before bind, and the
request path retains at most one socket and one bounded request owner.

## OpenAI-compatible HTTP/SSE server

`LunaOnlineOpenAIServer` is the distinct reusable HTTP/OpenAI owner. Its
preparation authenticates the model identity, inference limits, and maximum
transport wait from the same
`LunaOnlineFramedServicePreparation`; callers cannot substitute a parallel
model or deadline envelope. It preallocates one trusted request-receipt
workspace, HTTP/auth workspace, typed OpenAI inbound handoff workspace,
fixed HTTP response workspace, OpenAI semantic-event workspace, socket input,
and dual-view output scratch before listener ownership is published.

Each accepted socket begins a trusted Receipt and HTTP Work before the first
body read. HTTP request-line, header, bearer-auth, and body work retain that
original receipt. No semantic Stream is opened—and no Service admission
authority is mutated—until authentication, the complete HTTP body, and OpenAI
JSON/template conversion and semantic validation have produced a Ready typed
handoff View. At that point the exact Stream is opened and the handoff lease is
transferred with the same Receipt; request admission samples it once and
preserves its original absolute deadline. No canonical request frame is
rendered, copied, checksummed, or reparsed on this route. HTTP, OpenAI inbound,
typed admission, fixed-response, and OpenAI outbound
construction all advance cooperatively under their exact configured copy/work
quantum. Each charged transition checks the live Receipt or Stream stall
deadline before mutation.

OpenAI network-accept telemetry counts each socket committed to the HTTP owner,
even when authentication or validation later rejects it before a Stream opens;
the matching disconnect and finite network rejection are recorded exactly
once. `AdmissionRejected` is reserved for a typed framed preparation rejection
after Stream admission has actually been attempted.

WorkerFailure records both the worker incident and its request failure before
the recovered Usage/terminal tail can be lost to peer disconnect. The later
payload-free WorkerRequestTerminal observation closes request timing and
connection state without repeating either counter or structured log.

Pipeline boundaries are admission- and observation-authenticated. A fixed
counter increments only after an Admission observation is successfully
recorded and decrements only after Completion, Cancellation, Deadline,
RequestFailure, or WorkerRequestTerminal is successfully recorded. Parse and
admission rejection and cancellation before admission create no counter debt;
WorkerFailure is deliberately nonterminal; connection retirement clears any
remaining debt without publishing a connected boundary. Zero/overflow fail
closed before telemetry mutation, and a connected boundary additionally
requires exact coordinator quiescence, empty transport tail, and counter zero.

The server does not send `200 text/event-stream` merely because a frame was
received. It waits until the request has produced its first authenticated
semantic event and that event's OpenAI compatibility View is Ready. A
pre-header validation, admission, deadline, capacity, or service rejection is
returned as one fixed payload-safe 400/401/429/503 response. Once the 200 head
is fully confirmed, later failure is fail-close; a second HTTP response is
never appended to an SSE stream.

For every response or SSE fragment, the owner copies no more than the relevant
HTTP/OpenAI work quantum and configured socket scratch capacity. A positive
`write_once` count advances the exact source cursor before the immutable Flight
is released. Zero, timeout, cancellation, or unknown completion confirms no
bytes. Only after the entire compatibility View is confirmed does the server
release it and call the exact semantic event's `delivered`; later cooperative
Stream progress performs the lower ACK. Usage-to-terminal replacement therefore
cannot occur before Usage bytes are delivered, and Completed/Failed marks the
connection terminal only after its final SSE bytes (including `[DONE]`) are
confirmed.

One HTTP request is accepted per connection. Pipelined or trailing bytes are a
validation error before stream admission, not a second request. Connection
retirement revokes retained Receipt, HTTP, frame, response, semantic, rejection,
observation, Flight, and Stream authority before the same listener/Service can
accept the next connection. Service drain remains listener-first and is driven
cooperatively on the same serialized runtime task. The additive server owns no
raw HTTP View, canonical frame, Luna event View, socket, buffer, or epoch escape.

## One-shot compatibility endpoint

The compatibility `LunaOnlineTcpEndpoint` owns a bounded, serialized, native TCP shell around one
`LunaOnlineFramedCoordinator`. It internally binds one `TcpServer` from an
immutable address, accepts exactly one connection, closes the listener, and
retains the connection and coordinator in one opaque endpoint through cleanup.
It is not a reusable listener, a multi-client server, an HTTP adapter, or a
standalone reactor/runtime.

The caller retains the same `LunaOnlineTcpEndpoint` while accept is pending and
after every timeout, I/O failure, or cancellation. No accept result transfers
socket authority. All buffers, the listener, accepted socket, coordinator,
framed Offer, and scratch Flight remain private. Mutually exclusive activity
guards reject retained endpoint aliases that try to overlap reactor progress
with another reactor or off-reactor maintenance call.

Those activity flags are lifecycle authentication, not atomic locks. The host
must serialize handoff and never access the endpoint simultaneously from a
reactor and maintenance OS thread; the guards detect retained aliases only
inside that exclusive-access contract.

`progress_on_reactor` is the only method that accepts, reads, writes, or closes
a socket. It runs at most the configured number of coordinator/transport
transitions and performs at most one bounded async socket operation per call.
`progress_off_reactor_maintenance` is synchronous and never touches a socket;
when inner maintenance completes it publishes `CloseRequired`, and the host
must call reactor progress to close the connection. LunaFlux still requires a
host-owned executor to place that maintenance call off the socket reactor.

Ingress uses one fixed startup buffer. A short coordinator acceptance retains
the exact unread tail and no further socket read occurs until the tail is
consumed. Zero consumption is backpressure, not EOF. Socket EOF, accept/read/
write timeout, OS error, and cancellation are terminal for this one-shot
endpoint. Cancellation cleanup closes any committed socket/listener and
disconnects the coordinator before propagating the cancellation error.
An explicit coordinator `AwaitingInput` result starts the next bounded read
when no retained tail remains; `Advanced` never implies that the socket should
be read. This prevents a fragmented prefix from spinning inside the reactor
transition budget.

For output, the endpoint copies one bounded coordinator event chunk into the
private scratch, publishes an authenticated Flight, and calls `write_once` on
the immutable view. A positive short write confirms that exact opaque Offer
before releasing the Flight; the remaining event bytes are recopied under a
new Offer. Zero progress, timeout, cancellation, or unknown write completion
never confirms bytes and instead disconnects and drains. Usage-event ACK work
remains in off-reactor maintenance.

A preparation rejection is protocol-terminal in this first shell. The endpoint
reads its bounded sequence and rule, ACKs the exact rejection credit, records
only those scalars, and closes without inventing a Luna event frame.
`Rejected` is an observable disposition, not a closed state: especially when
it is returned by off-reactor maintenance, the socket remains `CloseRequired`
and the host must call reactor progress before continuing drain/cleanup.

`LunaOnlineTcpOutputScratch` allocates one dynamic `Bytes` backing at startup.
The narrow native bridge in `internal/online_tcp_buffer_alias` gives that same
MoonBit object a mutable `FixedArray[Byte]` view; the C return increments the
object reference count so the two stored MoonBit references are balanced
independently. The bridge is one-way and receives no literal, foreign, or
externally supplied buffer. Neither raw view nor a scratch capability is
public from this service package.

Startup rejects nonpositive read/write capacity and any capacity above
16,777,216 bytes before allocation. Timeouts and the transition quantum are
also explicitly bounded and are defensively revalidated by bind before any
buffer, owner, or socket allocation. Bind first authenticates the retained
coordinator preparation's own maximum transport wait and rejects input-idle or
write waits above it; a caller cannot substitute a longer inference envelope.
Immediately before each body read or event write, the endpoint queries the
live coordinator's shorter current receipt/stall remainder. It waits for at
most the configured interval or `remaining - 1`; a remainder of one
millisecond or less fails closed without starting another socket operation.

Only an exact-generation write capability may mutate the backing. Publishing
moves the scratch into write-in-flight state, where retained write aliases are
stale and only the matching flight may inspect or release the bytes. Abort and
release return the scratch to idle; reuse advances a nonwrapping generation.
Per-operation methods allocate no new storage.

Async task/timer and OS socket operations may allocate; this package makes no
reactor-allocation-free claim. The narrower post-start scratch mutation,
publication, inspection, and release operations retain their existing
allocation-free generated-C evidence.
