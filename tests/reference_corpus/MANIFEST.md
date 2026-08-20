# External reference corpus

This corpus pins the Apache-2.0 upstream test model
[`stas/tiny-random-llama-2`](https://huggingface.co/stas/tiny-random-llama-2)
at immutable revision `3579d71fd57e04f5a364d824d3a2ec3e913dbb67`.
The exact 210,712-byte BF16 safetensors file is checked in because it is a
genuinely tiny functional-test artifact and is required for an offline,
network-free end-to-end correctness gate.

## Artifact identities

| Upstream file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 680 | `9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622` |
| `tokenizer.json` | 64,223 | `0afe36ee1358ce1fa277f4eac935250bb90253ed5c27867eb6ff376ded7d1980` |
| `tokenizer.model` | 499,723 | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` |
| `model.safetensors` | 210,712 | `852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30` |

The checked-in `config.json` and `model.safetensors` are exact upstream bytes.
`tokenizer_semantics_probe.json` is a deliberately compact derivative retaining
the upstream normalizer,
pre-tokenizer, post-processor, decoder, unknown-token, and byte-fallback
semantics needed to prove that LunaFlux rejects this tokenizer. It is not
presented as the upstream tokenizer file.

Checked-in derivative identities:

| Fixture | Bytes | SHA-256 |
| --- | ---: | --- |
| `corpus.json` | 1,794 | `f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0` |
| `tokenizer_semantics_probe.json` | 796 | `4ac8db0bb211b5bd064b58eaab43a3bd38de593593eceadc7364848fcf4f954f` |
| `compatible_bytelevel_tokenizer.json` | 48,574 | `82000898a396651e5d3b149c5dec02998427406c78ccdf14084cbf85c63d2686` |

`compatible_bytelevel_tokenizer.json` is a synthetic functional-compatibility
fixture, not an upstream artifact, not a tokenizer trained for this model, and
not a SentencePiece approximation. It exists only to exercise LunaFlux's
supported exact ByteLevel-BPE subset against the model's dense 3,000-row
vocabulary. It derives the complete byte alphabet and original eight merges
from the separately pinned compatible tokenizer corpus, assigns `Luna` to model
token ID 1 through the ranked merges `L u`, `Lu n`, and `Lun a`, preserves `*`
and `c` as byte IDs 42 and 99, moves raw byte `0x01` to ID 266, and fills IDs
267 through 2999 with unique unmerged `<lf:NNNN>` marker pieces. Consequently
the exact functional mapping `Luna*c` to `[1, 42, 99]` reaches the independently
generated third model case without claiming the upstream tokenizer's
normalization, BOS-template, unknown-token, or byte-fallback semantics.

## Independent generation

Tokenizer IDs were generated from the upstream `tokenizer.json` with
`tokenizers==0.22.2`. Logits and four-token greedy continuations were generated
with `transformers==4.53.2` and CPU `torch==2.7.1`, loading the immutable BF16
safetensors artifact into Float32 evaluation and using `LlamaForCausalLM` with
no cache. The commands ran outside LunaFlux; neither LunaFlux kernels nor its
executor produced expected values.

Each model case records last-position logits at selected vocabulary indices,
the full-vocabulary argmax, and greedy continuation. Comparisons use absolute
and relative tolerances of `1e-5`. This tolerance covers ordinary Float
accumulation-order differences while remaining far below observed argmax
margins.

## Compatibility result

The model is a compatible dense Llama artifact: BF16, untied embeddings, two
layers, all 21 exact LunaFlux tensor names, and admitted dimensions. The
tokenizer is intentionally **not** compatible with LunaFlux's selected
ByteLevel-BPE subset: it uses prepend/replace normalization, template BOS
insertion, byte fallback, unknown-token fusion, and no ByteLevel pre-tokenizer.
Therefore model cases use explicit token IDs and the tokenizer observations are
provenance evidence, not a claim that LunaFlux can encode this tokenizer.

The black-box MoonBit tests verify all artifact digests through the production
approved-root loader, then run complete LunaFlux admission, exact binding, host
materialization, text encoding, reference execution, sampled-logit comparison,
and greedy generation. The upstream tokenizer compatibility result remains
explicitly negative rather than being silently approximated; the synthetic
tokenizer proves only the supported functional text-to-token-to-model path.
