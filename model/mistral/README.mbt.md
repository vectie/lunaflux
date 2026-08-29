# Bounded Mistral model plan

This package ends the `MistralForCausalLM` family branch at immutable model-plan
construction. It emits only the existing dense decoder operations and kernel
capabilities; scheduler, KV, worker, device, and kernel packages never inspect
the family.

Mistral sliding attention is equivalent to full causal attention only while a
request remains within one sliding window. The admitted semantic spec therefore
caps its context at `min(max_position_embeddings, sliding_window)`. A null
window retains the full position ceiling. No longer request is silently routed
through the full-attention kernels.

The sibling `model/mistral_weights` package now owns the family-specific tensor
translation and delegates authenticated file inspection and materialization to
the generic numeric-weight owners. This package remains plan-only. Neither
package claims a Mistral startup route, physical numerical agreement, accuracy,
memory, performance, or production readiness.
