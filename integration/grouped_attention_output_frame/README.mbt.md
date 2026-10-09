# Prepared grouped attention output

Inverse suffix rotation, group-local low-rank projection and block-FP8 output
projection contribute three launches to the caller's queue. Weights stay compact;
Output-A dequantizes each parameter with a BF16 boundary before multiplication,
while Output-B uses per-block activation quantization. Preparation checks every
borrowed span before acquiring scratch, allocates three bounded outputs and
submits nothing. Close the borrowing queue before the frame and its operands.
