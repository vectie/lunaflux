# GLM-5.3 checkpoint manifests

This package deterministically enumerates the exact ordered logical tensor
names, dtypes, and shapes for GLM-5.3 full and GLM-5.3-Flash-BF16. Manifests
bind the authenticated model content and GLM execution-plan identity into a
canonical layout digest. Startup binding rejects missing, extra, reordered,
duplicate, dtype-shifted, or shape-shifted entries before payload access.

Flash control tokens are embedding rows, not independent checkpoint tensors;
their six IDs are therefore carried by the manifest's control-token contract
and validated against the fixed vocabulary.

The shared safetensors v1 materializer currently accepts BF16 tensors only.
Official GLM checkpoints also contain required F32 router/recurrent/mHC state,
and the full release uses FP8 block scales with a per-tensor exclusion list.
Those physical paths remain explicitly fail-closed until the shared dtype and
materialization contracts represent them. The full manifest currently covers
the exact 59,585 non-scale logical tensors of a BF16-converted artifact; it
does not claim the 59,044 FP8 scale tensors are executable.

The nvfp4_expert_weights adapter maps Flash routed expert partitions (layers
3..44 and MTP layer 45) to the installed ModelOpt NVFP4 checkpoint's gate/up/down
planes: packed U8 payload, E4M3 block scale for 16 input columns, and an F32 scalar
global scale. It uses the shared precision representation and leaves the
logical graph unchanged. Shared experts, dense layers and attention weights
are not reclassified as NVFP4.

The packed upload integration streams these planes directly into caller-owned
device buffers, without a full host bank or resident BF16 duplicate. Native
tests cover an official-sized expert through this data path; full GLM serving
and multi-node expert execution are not yet demonstrated by this loader work.

shared_expert_weights maps the installed Flash checkpoint's replicated shared
gate/up/down matrices to dense BF16 with no scale sidecars. Gate/up are
2048-by-4096; down is 4096-by-2048. These shapes and dtypes were checked against
the installed safetensors headers for layers 3, 44 and 45. The shared upload
adapter streams them into the generic compact expert bank and uses the same
precision IR and ordered program as routed experts, with expert ID 0 and unit
routing score. This does not change the routed NVFP4 representation.
