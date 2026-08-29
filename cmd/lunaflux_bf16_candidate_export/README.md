# Offline BF16 candidate export command

This narrow native command authenticates one dense-Llama configuration beneath
an approved model root, derives the canonical single-row paged `ModelPlan`,
closes the root, builds the exact sm120 BF16 candidate set, and atomically
publishes its offline compiler inputs. Bounded compiler major, minor, and patch
components are explicit and are bound with the independently supplied
toolchain-manifest SHA-256 into every recipe.
The command never invokes a compiler or opens process, device, runtime, worker,
or serving authority.

```sh
moon run --target native cmd/lunaflux_bf16_candidate_export -- \
  MODEL_ROOT CONFIG_LOCATOR CONFIG_SHA256 MODEL_CONTENT_SHA256 \
  TOOLCHAIN_SHA256 COMPILER_MAJOR COMPILER_MINOR COMPILER_PATCH \
  TOKENS_PER_PAGE TOTAL_PAGE_COUNT \
  MAX_QUERY_TOKENS MAX_PAGE_TABLE_ENTRIES ABSOLUTE_NEW_OUTPUT
```

For the checked-in approved tiny BF16 input, use `tests/reference_corpus` as
`MODEL_ROOT`, `config.json`, config digest
`9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622`,
model-content digest
`852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30`,
CUDA 13.1.115 uses `13 1 115`, followed by `8 32 1 32` for the four
capacity arguments. The emitted
`candidate_root` and digest-suffixed `candidate_inventory` values are direct
inputs to `scripts/build-luna-bf16-kernel-set.sh`.

The command output fixes `compiler_invoked`, `device_opened`,
`runtime_authority`, and `physical_readiness` to zero. A separately approved
CUDA build host must still compile and authenticate the candidate set.
