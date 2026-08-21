# Private Luna online TCP scratch

This package currently owns only the private dual-view output scratch required
by a future serialized native TCP shell. It does not import async, open a
listener, own a socket, or claim network ingress.

`LunaOnlineTcpOutputScratch` allocates one dynamic `Bytes` backing at startup.
The narrow native bridge in `internal/online_tcp_buffer_alias` gives that same
MoonBit object a mutable `FixedArray[Byte]` view; the C return increments the
object reference count so the two stored MoonBit references are balanced
independently. The bridge is one-way and receives no literal, foreign, or
externally supplied buffer. Neither raw view nor a scratch capability is
public from this service package.

Startup rejects nonpositive capacity and any capacity above 16,777,216 bytes
before allocation. That private ceiling is removed when the TCP constructor
can derive the exact event-frame capacity from validated framing limits.

Only an exact-generation write capability may mutate the backing. Publishing
moves the scratch into write-in-flight state, where retained write aliases are
stale and only the matching flight may inspect or release the bytes. Abort and
release return the scratch to idle; reuse advances a nonwrapping generation.
Per-operation methods allocate no new storage.

The future TCP slice must keep socket writes inside this package rather than
expose either raw view. It must also bind partial-write confirmation to the
authenticated flight lifecycle before adding a listener or connection owner.
