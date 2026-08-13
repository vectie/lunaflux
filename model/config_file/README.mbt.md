# Approved model configuration file admission

This synchronous startup package reads one typed relative locator beneath a
caller-owned `ApprovedRoot`. It obtains one bounded immutable snapshot, closes
the file before publication, verifies an independently supplied lowercase
SHA-256 identity, and only then delegates semantic JSON admission to
`model/config_reader`.

The independently supplied model `ContentDigest` binds the resulting
`LlamaModelMetadata`; it is deliberately not inferred from the configuration
file digest. Errors retain no locator, file bytes, native descriptor, errno, or
attacker-controlled JSON value. The package exposes no async, string-root,
ambient filesystem, or raw-handle API.
