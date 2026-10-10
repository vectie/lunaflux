# Offline serial decoder prompt frames

`encode_prompt` converts already-tokenized IDs into bounded canonical wire
chunks for a single contiguous diagnostic request. The last chunk is final
prefill; earlier chunks do not sample. It has no model-family, CUDA, network,
filesystem or tokenizer dependency. All allocation and rendering are offline,
not in the generation loop. This is not authority for live scheduler page
ownership; the serial diagnostic runner owns its request-local retained state.

The producer reserves at least one context position for generation, validates
token IDs against the supplied envelope, and retires each temporary plan only
after copying the frame into owned immutable bytes.

`PreparedSerialPrompt` resolves an entire generation request before GPU startup.
It requires contiguous positions from zero, one request identity, ordered model
generation/sequence fields, and exactly one final producing chunk. Prompt length
plus the requested maximum generation must fit the reserved context. Encoded
frames have an aggregate retained-byte ceiling and are immutable snapshots;
changing a caller's frame list cannot change the prepared replay. There is no
checksum scan, model-family policy or hot-path filesystem access.

The GLM and DeepSeek diagnostic generation adapters load each frame once before
device preparation, close input-file authority, and reuse these snapshots.
Their `preflight-generation` commands perform the same CPU-only planning before
a remote peer begins weight loading. This is not live multi-request serving.
