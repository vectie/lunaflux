# Prepared dependent categorical chain

The shared `SequentialMarkovPrecision` describes numerical and dependency laws,
not a model family or CUDA schedule. Each row gathers the previous sampled token's
BF16 embedding, projects a BF16 matrix in F32, adds its vocabulary bias to that
row's F32 logits, and samples the next token. Confidence is an F32 dot over the
pre-normalization BF16 hidden row and that same previous-token embedding. No
sigmoid or normalized-hidden substitution is permitted.

The CUDA backend lowers this into four ordered launches per row. Zero temperature
uses stable argmax with lowest-index ties. Positive temperature uses categorical
Gumbel-max with explicit seed and sequence device ports. The distribution matches
categorical temperature sampling; random samples are not claimed seed-identical
to PyTorch's independent generator. A caller must advance the sequence for a new
prediction and restore logits before replaying the chain: bias addition is an
explicit mutation, not an idempotent transform.

The prepared frame owns only embedding scratch, an immutable row-index plane and
function handles. All weights, counts, hidden/logits and result/RNG ports are
borrowed. Tables and argument storage are prepared once; the parent queue owns
submission, completion and cancellation. Budget, spans, context and writable-port
alias checks precede allocation. Closing the borrowing queue precedes closing the
frame. Component tests/compilation do not prove whole-model prediction.
