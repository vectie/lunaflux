# Decoder routing CUDA source

This family-neutral package renders deterministic correctness-first CUDA source
for two distinct routing stages: biased stable top-k index selection, and
selected-score gathering with optional ordered F32 normalization and scaling.
The finalizer accepts indices from either stable top-k or token-hash lookup.

Equal choice scores select lower expert ids. Official `torch.topk` does not
guarantee stable tied indices, so tie parity remains an explicit qualification
limitation. Source results grant no compiler, artifact, loader, runtime launch,
or qualification authority.
