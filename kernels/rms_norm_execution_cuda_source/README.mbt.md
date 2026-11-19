# RMS normalization source

Typed-IR lowering for BF16 input, weight and output with FP32 arithmetic. One
CTA owns each vector; the initial square reduction is ordered on thread zero.
Inputs can be strided projection views, outputs are packed. Gate inputs use the
packed output layout. The weight vector is shared across rows and heads.

The ABI is `counts, input, weight, [gate], output`; `counts[3]` is the active row
count. The launch is `(maximum_rows, vectors_per_row, 128)`. Select direct or
centered gamma, or the staged BF16 norm/weight plus SiLU gate, through the plan.

Source generation is not CUDA compilation, numerical qualification or a
performance result. No device handles or model-family branches are exposed.
