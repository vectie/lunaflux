# Recurrent physical planning

`ChunkDeltaPlan` refines an explicit `ChunkTriangularFloat32` numerical policy.
It owns frame-local factor sizes, independent value tiles and the ordered
prepare → solve → output/cache-commit effects. Sequence ordinals own scratch;
persistent request slots own state. No CUDA, vendor or model names occur here.
F32 triangular evaluation is not an exact rewrite of ordered token recurrence.

The initial lowering supports up to 64 frame rows and 256 components. A larger
frame is rejected for this explicit policy; the separate ordered policy retains
its existing capability. All sizes contribute to the parent workspace ceiling.
Comparison and empirical acceptance belong to external tests, not this plan.
