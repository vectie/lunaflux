# Qwen3 token-ID streaming bridge

This package is the Qwen-only streaming adapter used by the external
LunaFlux/vLLM/SGLang comparison. Each token event carries the exact native
output token ID and the corresponding stable incremental text fragment.

The bridge executable authenticates and parses the pinned `tokenizer.json`
once before listener readiness. Request processing reuses that immutable
`TokenizerSpec`; it does not read the tokenizer file, hash artifacts, or parse
JSON tokenizer state during generation.

Incremental decode preserves the admitted Qwen ByteLevel-BPE pieces. It uses
the same default baseline policy: tokenizer-special tokens are skipped while
ordinary added tokens remain visible. UTF-8 scalars split across token
boundaries are held until complete. Invalid maximal byte subsequences become
U+FFFD, and a trailing incomplete scalar is carried by the terminal event's
`text` field. Token events remain one-for-one with native output tokens, so
text stabilization never changes token timing or token identity.

The response is SSE. Token events use
`lunaflux.benchmark-token.v1`; the final event uses
`lunaflux.benchmark-terminal.v1` and is followed by `[DONE]`. Consumers must
append `text` from both token and terminal events to reconstruct the complete
decoded output.
