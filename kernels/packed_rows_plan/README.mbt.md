# Explicit packed row assembly

Pure source-to-destination row mapping for three already-projected matrices.
The caller supplies conditioning, video and audio positions in source-row
order. Destinations must be unique across all sources; unassigned rows are
zero padding. There is no assumed modality ordering or device policy.

Preparation snapshots positions and encodes a little-endian pair per output
row: source tag (0 padding, 1 conditioning, 2 video, 3 audio) and source index.
Upload once, keep immutable during execution. Width and an explicit total
element budget bound construction. This is a layout primitive, not a builder
for model-specific positions, timestep indices, rotary coordinates or masks.
