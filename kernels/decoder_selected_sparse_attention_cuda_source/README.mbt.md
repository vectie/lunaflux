# Decoder selected sparse-attention CUDA source

This family-neutral package renders an inert correctness CUDA source for
selected-index attention over shared BF16 K/V storage. The eight-operand ABI
accepts already joined indices and borrowed K/V storage; index construction,
K/V preparation and ownership, inverse RoPE, output projection, compilation,
qualification, and execution authority remain outside the package.

The numerical contract uses serial ordered F32 dot products, stable softmax,
and one BF16-RNE output boundary. The per-head F32 attention sink contributes
only to the denominator and never to the value numerator.
