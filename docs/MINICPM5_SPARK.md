# MiniCPM5-2B compatibility fixture (not the deployment model)

The user selected the strongest MiniCPM robot model; deployment now targets
[MiniCPM-V 4.7 35B-A3B](MINICPM_V47_SPARK.md). The small model below is retained
as a tested compatibility fixture and does not satisfy that selected-model gate.

The text-only compatibility fixture is `openbmb/MiniCPM5-2B`, pinned to
revision `42433e90dad95123ab87729fd7011e7110629556`. This pin identifies an upstream
revision; existing LunaFlux content-label fields do not verify downloaded bytes.
Deployment owns artifact acquisition and immutability.

The [upstream configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/42433e90dad95123ab87729fd7011e7110629556/config.json)
declares `LlamaForCausalLM`: 42 layers, hidden 2048, intermediate 6144, 16 query
heads, two KV heads, head width 128, vocabulary 130560, BF16, untied embeddings,
SiLU, RMSNorm epsilon `1e-6`, and unscaled RoPE theta 5000000. The existing dense
Llama spec/plan owns these semantics. No second decoder implementation is added.

Software implemented:

- Config admission accepts the EOS metadata array `[1,130073]` and rejects invalid
  ID arrays; EOS policy is still explicit at serving admission.
- Tokenizer admission recognizes the exact ordered numeric/word/ByteLevel splits,
  identity normalization, added tokens, complete byte alphabet, and BOS-zero
  TemplateProcessing. The numeric pre-split plus word split reuses digit-triplet
  execution. A declared unknown token is unreachable with the complete byte
  alphabet; unsupported fallback/merge behavior remains rejected.
- `integration/minicpm5.chat_template` supplies the bounded text-only, no-tools,
  no-thinking-history chat template. The tokenizer inserts BOS once, so the chat
  document prefix is empty. The assistant generation cue closes an empty think
  block, matching upstream `enable_thinking=false`.
- `integration/minicpm5.stop_token_ids` returns `[1,130073]`. Use those explicit
  stops in the existing inference policy and keep `</s>`/turn-end out of the
  generated skill JSON.

Focused config/tokenizer tests pass 90/90 locally. The pinned real metadata also
builds a paged Llama plan, admits all 130560 vocabulary entries, and passes text,
Chinese, numeric, emoji, added-token and chat-boundary checks (1/1 asset test).
These are parsing/round-trip/plan checks, not independent upstream-tokenizer
equivalence, reference-logit or model-quality evidence. This work adds no CUDA
kernel or native ABI change and carries no physical Spark execution claim.

From MoonRobo, prepare metadata without downloading model weights:

```sh
moon run scripts/prepare_minicpm5_assets.mbtx NEW_ASSET_DIRECTORY
```

From LunaFlux:

```sh
LUNA_MINICPM5_ASSET_DIR=ABSOLUTE_ASSET_DIRECTORY moon test --target native --warn-list '-79' tests/minicpm5_assets
moon test --target native --warn-list '-79' model/config_reader tokenizer tokenizer/json_reader integration/minicpm5
```

The local compiler reports existing implicit-method warnings (79) in dependencies.
Focused warning-denied check additionally needs existing warning exclusions
`-79-29-25-20`; none are global module configuration changes. New code is formatted
and generated interfaces are refreshed. Unrelated working-tree changes are retained.

Remaining Spark integration: acquire the pinned weights on the target, bind the
existing Llama tensor/materialization and kernel preparation paths to this actual
model shape, select the text chat profile at instance startup, and qualify exact
GB10 `sm121` execution. Start with a bounded 4096-token service context and 256 new
tokens for proposals; the model's 131072 context ceiling is not a KV allocation
recommendation. Verify upstream tokenizer IDs, teacher-forced logits, greedy
continuation, both EOS stops, cancellation, resource balance and JSON skill quality.
Do not transfer earlier `sm120` or other-model evidence to MiniCPM on Spark.

MoonRobo's `policy_planner` consumes the outer chat API and sends proposals to the
Atlas task service. LunaFlux imports no robot, skill supervisor or actor runtime.
See the sibling [two-host runbook](../../moonrobo/deploy/lunakine/README.md).
