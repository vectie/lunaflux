# Decoder compressed-index CUDA source

This family-neutral package renders deterministic I32 CUDA source for causal
complete compressed-cache slot selection. Its ABI uses explicit query-position
and cache-offset inputs and returns one count plus a `-1`-padded dense slot row.

The package owns source construction only. It provides no compiled artifact,
loader, launch, device-memory, scheduler, KV-cache, or qualification authority.
