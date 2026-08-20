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

`TextInput` retains the canonical immutable UTF-8 bytes accepted at the
contract boundary, so bounded Luna tokenization does not recreate them from
MoonBit's UTF-16 `String`. `TokenBuffer` is opaque. Ordinary constructors own
an immutable copied collection; startup-preallocated
`LunaTokenBufferStorage` instead publishes an exact-generation lease whose
tokens are readable only through scalar `token_status`. Releasing or reusing
the storage makes every retained buffer alias stale, and Luna-leased buffers
never expose a collection view or mutable fixed storage.
