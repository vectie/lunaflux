# MiniCPM-V 4.7 decoder stack

This package connects the prepared recurrent/full-attention and MoE branches
through the selected decoder's two centered RMS norms and two BF16 residual
updates per layer. One mutable hidden bank, normalized-input bank and branch
output bank are reused across all layers. The plan assigns disjoint persistent
state slices in model order. `integration/minicpm_v47_text` wraps this stack
with the existing embedding/final-norm/logit/greedy owners. Vision and serving
remain separate integration work.

The MoE workspace retains a startup-uploaded expert map. It therefore has its
own shared arena; attention and recurrence share another arena. Reusing the
MoE arena for those branches would corrupt that map. Each arena is reused
serially across all layers, never multiplied by the layer count.

The frame owns child frames, common function handles and the complete ordered
prefill/decode launch descriptions. Modules, weights, metadata, activations and
state arenas are borrowed. Close enclosing executors before the frame, then
release the borrowed resources. Close partially prepared frames on failure.
CPU source/layout tests do not establish CUDA compilation or execution.

At 32 rows, the three activation banks occupy 131,072 bytes each. Shared branch
scratch is 1,845,248 bytes and shared MoE scratch is 1,756,224 bytes. For the
initial single-sequence 4096-token cache, the four persistent arenas occupy
1,474,560 bytes of convolution history, 62,914,560 bytes of recurrent matrices,
and 41,943,040 bytes per KV plane. Each phase has 920 ordered launches.
The software ABI fixture exercises preparation, complete queues and explicit
cleanup through the real native wrappers with the existing test double.
