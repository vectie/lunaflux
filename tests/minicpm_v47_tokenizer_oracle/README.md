# Independent flagship tokenizer comparison

This optional offline test loads the actual selected model's `tokenizer.json`
with the upstream [Hugging Face tokenizers 0.23.2](https://github.com/huggingface/tokenizers/tree/v0.23.2/tokenizers)
Rust library. The small reference adapter only calls encode with automatic
special tokens disabled and decode with special tokens preserved. Cargo pins
the upstream version and the checked-in lockfile pins its dependency graph.
There is no runtime dependency on Rust, Hugging Face or Python in LunaFlux.

MoonBit constructs the corpus and compares every token ID and decoded string.
Cases cover multilingual robot instructions, NFC decomposition/reordering and
Hangul composition, emoji/ZWJ sequences, digits, whitespace/control characters,
contractions, skill JSON, literal/special markers, pairwise word/separator
combinations, optional system prompts, and generated image placeholders.
Duplicate input strings are removed before comparison.

```sh
moon run scripts/verify-minicpm-v47-tokenizer.mbtx ABSOLUTE_REPO PINNED_ASSET_DIRECTORY NEW_OUTPUT
```

Cargo and a native Rust/C toolchain are needed for this test only. The first
build downloads the locked dependencies; it uses the repository's ignored
`target/minicpm-v47-tokenizer-rust-build` directory. Output includes the exact
corpus, upstream observations and a scoped report with the declared asset
revision. Current local evidence passes 1,797 distinct strings and 28,043 tokens.

The prompts are rendered by LunaFlux before both tokenizers consume them. This
does not compare upstream Jinja rendering, unsupported history/tools/thinking
profiles, all Unicode inputs, logits, GPU execution or model quality.
