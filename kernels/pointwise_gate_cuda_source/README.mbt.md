# BF16 pointwise sigmoid gate

Packed input/output values multiply a BF16 sigmoid of a possibly strided gate
view. Both sigmoid and product round to BF16. The shared expert consumes the
same generic arithmetic law; recurrent SiLU-gated normalization uses different
FP32 stages and does not consume it.

ABI: `counts, input, raw_gate_projection, output`, with active rows in `counts[3]`.
The selected attention plan uses vector width 256, 16 vectors, gate vector stride
512, row stride 8192 and offset 256, reading each head's second half directly.
No gate-copy pass or interpretation of the query half is needed.

The plan and source generator own layout/arithmetic only. A prepared device
frame and target numerical/resource checks remain separate work.
