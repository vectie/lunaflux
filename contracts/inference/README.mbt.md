# Native inference contracts

This package owns the canonical transport-independent v1 generation request
and streaming event vocabulary. It validates every externally controlled
count and byte sequence before taking an immutable copy.

The package does not parse HTTP or compatibility JSON, tokenize text, admit or
schedule work, manage caches, or execute a model. Adapters translate into
these types; the scheduler consumes them without importing an API package.
The selected loaded-model identity is the canonical content-plus-plan identity
owned by `model/spec`; this package does not duplicate or reinterpret it.

`DeadlineBudget` is a relative client policy. The first trusted request-receipt
boundary converts it exactly once into an opaque monotonic
`AdmissionDeadline`; parsing, tokenization, mailbox, and scheduler delay all
consume that same budget. Scheduler admission accepts only the absolute value
and never rebases it from a later clock sample.

Payload-bearing request text, stop strings, decoded deltas, cache scopes, and
trace correlation deliberately have no derived debug representation. Public
validation errors contain only a bounded field, issue category, and collection
index.

Validated request-authority records are opaque outside this package. Callers
construct protocol and request identities, limits, sampling, stop, deadline,
cache, trace, and effective-limit values only through their checked public
constructors and typed accessors. `GenerateRequest::new` defensively
revalidates every nested request authority in a fixed precedence order, so a
same-package forged nested value passed to that constructor cannot bypass
token, duplicate, UTF-8 byte, scalar, or safe-identifier bounds.
`RequestGeneration` remains a separate monotonic
coordination value rather than validated request content; public output/event
records are also outside this request-authority boundary.

Input revalidation authenticates the exact owned or leased token backing,
rejects empty and negative token data, and recomputes the cached maximum token
ID before scheduler admission can trust it. Text input is re-decoded from its
retained UTF-8 bytes and must exactly match the retained MoonBit string.
Sampling seeds are revalidated as nonzero before stop conditions.

Deployment limits are themselves opaque authority and are defensively
revalidated before any request field is inspected; `Limit` therefore precedes
`ProtocolVersion` and every later request error. Forged stop and identifier
strings are scanned as UTF-16 before UTF-8 byte accounting, so lone or
misordered surrogates return bounded `InvalidFormat` errors while valid
supplementary pairs count as exactly four UTF-8 bytes.
Code-unit length is checked first because it is a lower bound on UTF-8 bytes;
definitely oversized forged strings therefore return `TooLong` without an
unbounded malformed-string scan.

`TextInput` retains the canonical immutable UTF-8 bytes accepted at the
contract boundary, so bounded Luna tokenization does not recreate them from
MoonBit's UTF-16 `String`. `TokenBuffer` is opaque. Ordinary constructors own
an immutable copied collection; startup-preallocated
`LunaTokenBufferStorage` instead publishes an exact-generation lease whose
tokens are readable only through scalar `token_status`. Releasing or reusing
the storage makes every retained buffer alias stale, and Luna-leased buffers
never expose a collection view or mutable fixed storage. Both owned and leased
buffers cache their actual maximum token ID while performing the existing
validation/write pass. `maximum_token_status` authenticates the exact live
generation before comparing that cache with a caller's bound; neither the
cached value nor the construction-time validation ceiling is exposed.

`LunaRequestSemanticStorage` is the transport-neutral, startup-preallocated
owner for stop-token IDs, stop-string UTF-8, cache-scope bytes, cache
permission, and the exact `InferenceLimits` under which they were admitted.
Its generation-authenticated lifecycle is `Storage -> Write -> Work -> Lease ->
View`. The builder layer immediately enforces structural order and envelope:
token/string counts and cache length at `begin`, then each declared string
length before its bytes. It is intended to import an already structurally
validated framed view. A structurally complete write transfers to the semantic
layer, whose fixed precedence is token range/duplicates, string
empty/UTF-8/duplicates, then cache-scope safe grammar. This does not claim
multi-error equivalence with `StopConditions::new` or `GenerateRequest::new`
when a caller violates the builder envelope.

One charged semantic-work unit is exactly one token range check, prior-token
comparison, string header transition, UTF-8 byte or terminal check, prior
string length or byte comparison, cache header transition, cache byte check,
token-index copy, or insertion-sort comparison/move transition. Consequently
`progress` reports exact per-call and total units and never exceeds its step
budget. Semantic failure retains the exact generation and must be explicitly
aborted; it never silently returns storage to idle. Startup sizing reports the
exact `Int` and byte cells and checked arithmetic precedes every allocation.

`Lease` alone owns release. Every scalar or byte `View` read reauthenticates
the exact leased generation before checking its index, and no API exposes a
raw array, string, byte collection, storage index, or epoch. `is_stop_token`
performs a bounded binary search over the budgetedly constructed sorted index;
its allocation-free status distinguishes present, absent, and stale and
reports the exact number of token comparisons. Releasing the lease invalidates
all retained views immediately without clearing storage proportionally.
`LunaRequestSemanticLease::stop_token_view` narrows the same exact epoch to an
opaque scheduler projection with only allocation-free liveness and bounded
token-membership queries. It carries no stop-string, cache, limits, raw-token,
or release surface. A projection may issue at most two exact-ID
`LunaRequestStopTokenRetentionSlot` owners: one for the scheduler and one for
its online worker. `retain_into` binds directly into an empty startup-allocated
slot, so no copyable retention or boxed optional is created on admission. Each
slot exposes only liveness, bounded membership, and exact once-only release.
Semantic lease release rejects
transactionally with `LunaStorageBusy` while either retention remains, and a
copied retention alias becomes stale after the first release. Storage becomes
reusable only after both lower owners retire.
