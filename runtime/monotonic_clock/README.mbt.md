# Monotonic clock

`runtime/monotonic_clock` exposes one opaque, process-local clock capability.
`now_millis` returns `UInt64` elapsed milliseconds from an unspecified origin.
It is intended only for interval measurement and monotonic deadlines; it is
not civil time, a timestamp, or a wall-clock source.

The capability owns no native resource and needs no close operation. Its
implementation delegates to the private `internal/monotonic_clock` C ABI,
which calls only `clock_gettime(CLOCK_MONOTONIC)`, validates the returned
`timespec`, and checks the seconds-to-milliseconds conversion for overflow.
Native failures map to the bounded payload-free `Unavailable` or `OutOfRange`
errors; no `errno`, platform detail, clock origin, or time payload is exposed.

The package is cycle-neutral: it imports no contracts, scheduler, engine,
service, filesystem, process, or device package. Higher layers may accept the
opaque capability without moving native declarations outside their internal
owner.
