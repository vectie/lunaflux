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
