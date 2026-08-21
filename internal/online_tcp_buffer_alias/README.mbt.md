# Internal online TCP buffer alias ABI

This native-only internal package owns one representation bridge required by
the future serialized online TCP shell. It retains a dynamically allocated
MoonBit `Bytes` object and returns the same object as a mutable
`FixedArray[Byte]` view. The C stub increments the reference count because the
returned MoonBit value is a new owned reference to a borrowed argument.

The bridge is deliberately one-way. Its only admitted caller is
`service/online_tcp`, which passes the private dynamic `Bytes` created for its
startup scratch. Static byte literals, foreign buffers, and arbitrary external
inputs must never reach this primitive. No internal ABI type is exported.
