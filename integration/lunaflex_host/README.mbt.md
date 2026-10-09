# Optional LunaFlex host

This nested MoonBit module proves that LunaFlux and LunaFlex can coexist in one
native dependency graph without making LunaFlex a mandatory dependency of the
LunaFlux product module.

Run from this directory:

```sh
moon tree
moon check --target native --deny-warn
moon test --target native --deny-warn
```

The host checks LunaFlex ABI version 1 and the dense-BF16, single-CUDA-device,
continuous-batching, paged-KV, prefix-reuse, and AOT-execution capabilities.
Its black-box test also runs LunaFlex's tiny-Llama plan, materialization,
reference execution, sampling, and generation path while both modules are
linked in the same native workspace.

This is compile-time composition. Replacing LunaFlux's copied core packages
requires migrating one whole owning-package boundary at a time; MoonBit types
from `vectie/lunaflux/*` and `vectie/lunaflex/*` are nominally distinct and
must not be translated inside the token-step hot path.
