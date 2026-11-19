# MiniCPM-V 4.7 model specification

Owns the outer multimodal semantics of `openbmb/MiniCPM-V-4.7-35B-A3B`, revision
`98e79ba972b586ac6146d3505771ce1467e199ba`. The parser retains compressed vision,
canvas positions, marker IDs and the shared hybrid decoder geometry; it does not
turn this checkpoint into a dense Llama or the existing FP8 Qwen artifact.

`config.fixture.json` is the public configuration from that upstream revision.
Its omitted 2x2 window/merge settings and one final merge round follow
Transformers `v5.18.0`'s MiniCPM-V 4.7 defaults. Config parsing loads no Python,
weights or device runtime, and does not prove numerical execution.
