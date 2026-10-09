# DeepSeek V4 model-family boundary

This package owns startup-only DeepSeek-V4 semantic planning. It publishes the
complete capability contract and a canonical, typed advanced decoder plan with
exact model geometry, low-rank projections, YaRN position encoding, per-layer
hybrid attention, hyper-connections, routed/shared experts, MTP, and the
Flash-0731 DSpark attachment.

It intentionally does not lower V4 into LunaFlux's current dense generic
`ModelPlan`: doing so would erase hyper-connections, compressed/heavily
compressed attention, routed experts, and multi-token prediction. Backend
admission must satisfy all declared capabilities before executable lowering is
added. The architecture-neutral plan is owned by
`model/advanced_decoder_execution_plan`; the scheduler and KV packages remain
family-neutral.
