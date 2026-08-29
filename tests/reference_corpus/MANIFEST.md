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
`tokenizer_semantics_probe.json` is a deliberately incomplete compact
derivative retaining marker-shaped normalizer, pre-tokenizer, post-processor,
decoder, unknown-token, and byte-fallback structures. It proves that LunaFlux
rejects a nearby document that does not describe the complete closed selected
profile. It is not presented as the upstream tokenizer file.

Checked-in derivative identities:

| Fixture | Bytes | SHA-256 |
| --- | ---: | --- |
| `corpus.json` | 1,794 | `f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0` |
| `prefix_referee.json` | 792 | `765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3` |
| `tokenizer_semantics_probe.json` | 796 | `4ac8db0bb211b5bd064b58eaab43a3bd38de593593eceadc7364848fcf4f954f` |
| `compatible_bytelevel_tokenizer.json` | 48,574 | `82000898a396651e5d3b149c5dec02998427406c78ccdf14084cbf85c63d2686` |
| `upstream_tokenizer.json` | 64,224 | `fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b` |

`upstream_tokenizer.json` is the exact upstream JSON document plus one final
line-feed byte required by the checked-in text-file representation. The
independently approved digest above is therefore the LunaFlux artifact
identity; its parsed tokenizer semantics are byte-for-byte equivalent to the
upstream document while preserving fail-closed file authentication.

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

`prefix_referee.json` is a narrower derived campaign fixture. Its two greedy
tokens were computed separately for the eight-token full-page prompt and the
matching nine-token private-tail prompt by the approved scalar BF16 reference
executor after that executor passed the independently generated corpus above.
The spawned prefix requests never supply expected output to each other. This
fixture is independent of the device-worker and prefix-cache path, but it is
not represented as a fresh external Transformers logit capture.

## Compatibility result

The model is a compatible dense Llama artifact: BF16, untied embeddings, two
layers, all 21 exact LunaFlux tensor names, and admitted dimensions. LunaFlux
also admits the exact selected upstream `tokenizer.json` through a closed
SentencePiece-derived BPE profile: prepend/replace normalization, BOS template
insertion, complete byte fallback, fused-unknown policy, and the ordered
replace/byte-fallback/fuse/leading-strip decoder. Nearby or general
SentencePiece profiles remain fail-closed rather than being approximated.

The black-box MoonBit tests verify all artifact digests through the production
approved-root loader, compare the selected text corpus against independently
recorded upstream token IDs, then run complete LunaFlux admission, exact
binding, host materialization, tokenizer-produced empty-input tokens, reference
execution, sampled-logit comparison, and greedy generation. The synthetic
ByteLevel tokenizer remains separate compatibility evidence for its narrower
profile; it is no longer a substitute for selected-upstream tokenization.
