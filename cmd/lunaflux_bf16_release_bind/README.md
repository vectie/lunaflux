# Approved BF16 release binder

This native offline command re-admits the pinned tiny BF16 model, reconstructs
the exact `sm_120` candidate set for the final 8-token/32-page runtime shape,
authenticates every deterministic compiled-set receipt and CUBIN beneath a
no-follow approved root, and publishes a no-overwrite kernel-root source plan.
It then re-admits its own emitted schema-v2 manifest and derives the production
device-worker bootstrap digest through the same typed execution blueprint used
at runtime. It never invokes a compiler or opens a CUDA device.

```sh
moon run --target native cmd/lunaflux_bf16_release_bind -- \
  MODEL_ROOT COMPILED_SET_ROOT TOOLCHAIN_MANIFEST TOOLCHAIN_SHA256 \
  COMPILER_MAJOR COMPILER_MINOR COMPILER_PATCH ABSOLUTE_NEW_OUTPUT
```

The compiled set must come from a candidate export using capacity arguments
`8 32 1 32`. The output is the source consumed by
`scripts/assemble-luna-kernel-root.sh`; the printed
`admitted_bootstrap_sha256` is the exact value for the strict runtime
descriptor. Existing output or staging paths are never replaced.
