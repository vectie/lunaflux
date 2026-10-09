# Remote control stream

`RemoteChannel` owns an already-connected MoonBit native TCP socket, bounded
receive/transmit storage and one absolute local deadline per in-flight frame.
Call `begin_send`/`begin_receive`, then cooperatively poll both lanes and the
worker's lease on every turn, including idle turns. Close explicitly. Any EOF,
invalid length, clock regression, timeout or cancelled I/O closes the channel.
A closed channel cannot reconnect or admit another generation.

`poll_receive(..., allow_idle_eof=true)` optionally reports an orderly EOF before
any next-prefix byte as a closed channel with no frame. This opt-in does not
accept truncated prefixes/bodies, reset errors or cancellation. The stage server
uses it only after request release has been acknowledged, not after mere commit.

The length prefix is an unsigned 32-bit little-endian byte count. Frame bytes
are opaque; existing worker/rank-group codecs retain semantic and generation
validation. Partial reads never publish a partial destination, and a pending
write snapshots the source before yielding. Capacity is fixed at startup.

The transport does not authenticate a peer, supply encryption, or produce a
worker Ready capability. A deployment must supply an admitted control channel
before exchanging plans. The async socket lane is outside the scheduler;
this package does not claim allocation-free async task polling.
