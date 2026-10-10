# MiniMax checkpoint bootstrap

`ComponentCheckpoint::inspect` turns actual digest-declared component shards
into the complete semantic manifest consumed by `WeightStartup`. It retains
the original shard ordinals, validates every tensor and selects the existing
model-owned packings. `prepare` streams those packings directly into the
caller's aggregate device owner with bounded scratch, not a full host arena.

Text includes both text and vision weights; denoiser includes input, timestep,
context, refiners, all fifty blocks and output. Video and audio select complete
decoder packings. Reference-encoder execution is separately prepared; selecting
decoder packings does not claim support for an arbitrary reference request.

These are startup/file effects around immutable model/precision plans, not a
runtime JIT or model branching in the scheduler. Whole-request placement,
cross-host conditioning/latent handoff and real media generation remain required.
