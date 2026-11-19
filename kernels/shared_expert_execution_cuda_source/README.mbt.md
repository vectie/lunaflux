# Shared expert gate and routed combination

Lower typed numeric IR for the BF16 shared expert result and scalar gate logit.
The sigmoid rounds to BF16, multiplication rounds to BF16, and the sum with the
routed result rounds to BF16. The routed FP32 buffer carries the exact BF16 value
produced by the selected expert plan's `Bf16EachAdd` mode.

ABI: `counts, routed_f32, shared_bf16, gate_logits_bf16, output_bf16`. `counts[3]`
is active rows. One CTA owns a row and computes its gate once. Matrix projections
and shared SwiGLU remain separate existing operators. Generated source still
requires AOT compilation, GPU comparison and measured qualification.
