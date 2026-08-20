# Luna semantic event credit

`service/luna_event` owns the transport-independent, one-credit semantic event
slot used by Luna inference sessions. It does not know about sockets, HTTP,
framed encoding, schedulers, tokenizers, workers, clocks, or filesystems.

`LunaEventOwner` allocates its bounded payload storage once. The capacity is the
larger of the configured decoded-delta limit and the 64-byte public error-code
limit. Accepted, Token, Usage, Completed, and Failed publications then reuse
that storage without constructing `String`, `Bytes`, or `StreamEvent` payloads.

Each publication advances a non-wrapping owner epoch and exposes one
allocation-free `LunaEventView`. Its typed `accepted`, `token`, `usage`,
`completed`, and `failed` projections return opaque `#valtype` views only after
authenticating the exact live owner, epoch, and kind. Variant accessors repeat
that authentication before returning scalar evidence or copying bounded bytes
into caller-owned storage. Views expose neither mutable storage nor a raw
epoch.

Retirement requires the exact view borrowed from the owner. `discard` is a
separate abort operation: it invalidates a pinned view without treating a
consumer as having acknowledged it. A foreign, stale, wrong-kind, or retired
view cannot observe or retire a later event.

Terminal bundles use an atomic transition. `replace_usage_with_completed` and
`replace_usage_with_failed` authenticate the pinned Usage view, validate the
entire replacement and the next epoch, and only then publish a distinct epoch.
They inherit the exact Usage scalars, so a terminal event cannot substitute a
different accounting snapshot between credits.

`LunaEventPublishStatus` keeps the hot Token path allocation-free while
distinguishing invalid payload evidence, unavailable credit, and permanent
epoch exhaustion. Event epochs never reset across requests.

Before consuming a request, an aggregate can call the read-only
`LunaEventOwner::has_epoch_capacity` with its complete worst-case event count.
The check uses subtraction before conversion/addition, so it remains exact at
`UInt64` maximum without wrapping. A count of zero is valid at maximum; one is
not, and at `maximum - 3`, three publications fit while four do not. The query
does not reserve, retire, or otherwise mutate event credit.

This package owns semantic storage and read-only views only. The online
aggregate owns request-lifecycle acknowledgement authority, while an outer
adapter may stage a view into its transport format without receiving that
authority.
