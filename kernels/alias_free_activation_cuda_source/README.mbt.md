# Alias-free activation CUDA source

This family-neutral package renders the exact staged F32 alias-free activation
used by BigVGAN-style decoders: replicate-padded depthwise 2x transposed
convolution, SnakeBeta, then replicate-padded depthwise stride-2 convolution.
The upsample and downsample filters are distinct operands. The renderer grants
no compilation, loading, launch, or physical qualification authority.
