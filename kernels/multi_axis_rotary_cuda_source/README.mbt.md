# Multi-axis prefix rotary source

Lower a model-supplied frequency/axis map into a BF16 coefficient producer and
split-half prefix rotation. Unlike the existing suffix/adjacent-pair operator,
this plan rotates `[left half, right half]` in the prefix and copies the remaining
head words unchanged. It does not alter that existing operator.

The coefficient ABI is `counts, positions, cosine, sine`: positions are packed
int32 `[row,T/H/W]`, with model-supplied per-frequency axis choice and exact FP32
inverse-frequency bits. Cosine and sine each occupy `[row,rotary_width/2]` BF16.
Positions must satisfy the plan's nonnegative bounds. Named FP32 `cosf`/`sinf`
primitives round into BF16; CPU trigonometry is not a bitwise CUDA oracle.

The rotation ABI is `counts, input, cosine, sine, output`. Packed BF16 heads
round each product before their sum. Input/output may alias; both split halves
are read before either is published. Tail words copy bit-for-bit. `counts[3]`
is active rows. Query and key rotations can share the same coefficient planes.

Model builders own section/interleaving policy and frequency construction. This
package contains no model name, canvas parser, KV addressing or device owner.
Source generation is separate from CUDA compilation and GPU qualification.
