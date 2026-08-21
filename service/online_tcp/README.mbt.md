# Luna one-shot online TCP endpoint

This package owns a bounded, serialized, native TCP shell around one
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
