# Native inference contracts

This package owns the canonical transport-independent v1 generation request
and streaming event vocabulary. It validates every externally controlled
count and byte sequence before taking an immutable copy.

The package does not parse HTTP or compatibility JSON, tokenize text, admit or
schedule work, manage caches, or execute a model. Adapters translate into
these types; the scheduler consumes them without importing an API package.
The selected loaded-model identity is the canonical content-plus-plan identity
owned by `model/spec`; this package does not duplicate or reinterpret it.

Payload-bearing request text, stop strings, decoded deltas, cache scopes, and
trace correlation deliberately have no derived debug representation. Public
validation errors contain only a bounded field, issue category, and collection
index.
