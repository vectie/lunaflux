# FP8 v2 sm120 physical probe

This qualification-only executable exports the real production FP8 v2 QKV
and gated-MLP AOT CUDA sources for exact `sm_120`, then loads their offline
CUBINs through LunaFlux's public CUDA device wrapper. It compares repeated
launches with an independent ordered-F32 E4M3 referee and checks the production
finite-positive workspace scale policy.

The host readback and output poison are probe-only diagnostics. They do not
enter `ProductionFastPath`, runtime manifests, or worker execution. A passing
campaign remains non-bindable and does not itself grant readiness.
