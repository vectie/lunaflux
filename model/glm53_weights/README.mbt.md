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
