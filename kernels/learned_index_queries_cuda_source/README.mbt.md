# Learned index queries lowering

Composes existing block-128 E4M3/UE8M0 query projection, BF16 head projection,
head scaling and rotary/Hadamard/E2M1 renderers. Six prepared effects, not a
new queue. Input low rank is already normalized. Head weights round after
projection and after scaling; query transforms preserve their BF16 boundaries.
No sparse scoring, top-k or distributed reduction is implied by this source.
