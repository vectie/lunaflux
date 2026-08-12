# Compatible external tokenizer corpus

This directory pins a compact Apache-2.0 test derivative of Hugging Face
`tokenizers` at release `0.22.2`, commit
`f383101a26663708484cac0727792aad74f78234`. It exists to compare LunaFlux
against an implementation and expectations that were not produced by
LunaFlux.

Primary upstream sources:

- [`tokenizers/src/models/bpe/model.rs`](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/tokenizers/src/models/bpe/model.rs),
  test `test_tokenize_with_and_without_dropout`, supplies the `unrelated`
  vocabulary, merge order, and deterministic no-dropout result;
- [`tokenizers/src/pre_tokenizers/byte_level.rs`](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/tokenizers/src/pre_tokenizers/byte_level.rs),
  tests `pre_tokenization_no_regex` and `decoding`, supplies the no-regex
  ByteLevel transform and inverse; its `BYTES_CHAR` table defines the canonical
  256-byte alphabet;
- [upstream `LICENSE`](https://github.com/huggingface/tokenizers/blob/f383101a26663708484cac0727792aad74f78234/LICENSE)
  records the Apache-2.0 license.

`tokenizer.json` is a deliberately small source-derived fixture, not a claim
that the file occurs verbatim upstream. It combines the complete canonical
ByteLevel alphabet with the eight upstream merge rules and their result tokens.
Its pipeline is exactly LunaFlux's admitted subset: null normalizer and post
processor; ByteLevel pre-tokenizer and decoder with `add_prefix_space=false`,
`trim_offsets=true`, and `use_regex=false`; and BPE without dropout, unknown
fallback, byte fallback, unknown fusion, or ignored merges.

`corpus.json` was generated independently with CPython 3 and
`tokenizers==0.22.2` by loading that serialized tokenizer and calling
`encode(..., add_special_tokens=False)` and
`decode(..., skip_special_tokens=False)`. The cases cover ranked merges, spaces,
multi-byte Unicode, a literal zero byte, newlines, and Georgian text.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `tokenizer.json` | 2,783 | `72e19c379865733c37776d8386a87707ac50703aed1633fabb5c708b1479e142` |
| `corpus.json` | 1,369 | `c18cec0a5b3d2bbcab51d80153929f4ec6c81af91e5725a83508d07dfb67fbd9` |
| `corpus.json` | 1,369 | `c18cec0a5b3d2bbcab51d80153929f4ec6c81af91e5725a83508d07dfb67fbd9` |

The corpus is test evidence only. The production LunaFlux runtime does not
depend on Hugging Face or Python.
