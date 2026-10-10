# Literal checkpoint text input

Shared native frontend for installed tokenizer JSON and literal UTF-8 input.
This package owns no model family, chat template, weight loader or CUDA state.
Encoding rejects token overflow and adds no implicit BOS/EOS. The supplied
tokenizer label is not authenticated or recomputed. Input file handles close
before encoding; the caller retains ownership of its checkpoint root.

```mbt check
///|
test "literal text and encoder ID serialization" {
  let entries : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    entries.push({
      token_id: value,
      piece: Bytes::make(1, value.to_byte()),
      kind: Normal,
    })
  }
  let spec = @tokenizer.TokenizerSpec::new(
    entries,
    [],
    @tokenizer.TokenizerDigest::parse("a".repeat(64)),
    {
      max_input_bytes: 1024,
      max_output_tokens: 128,
      max_decoded_bytes: 1024,
      max_vocab_entries: 256,
      max_merge_rules: 1,
      max_token_bytes: 32,
      max_special_tokens: 1,
    },
  )
  let input = @checkpoint_text_input.encode_text_input(
    spec,
    b"a",
    maximum_tokens=1,
  )
  assert_eq(input.token_ids()[0], 97)
  assert_eq(input.decoded_bytes(), b"a")
  assert_eq(input.i32_le_bytes(), b"\x61\x00\x00\x00")
}
```

## Shared output decoding

This native frontend loads the installed tokenizer JSON for all checkpoint
diagnostics. It does not render model-specific chat templates, add BOS/EOS,
authenticate payloads, read weight shards or own CUDA resources. Caller-supplied
labels are compatibility metadata, not checksums.

`load_checkpoint_text` loads a tokenizer, encodes literal bytes and returns IDs
plus decoded bytes. `load_checkpoint_tokenizer` exposes the same bounded loader
for decoding generated output without inventing a second envelope. The returned
pure tokenizer owns no file or root handle; decoding is valid after root release.

`load_checkpoint_output_text` shares bounded comma-separated generated-ID parsing
and original-tokenizer decoding between GLM and DeepSeek diagnostics. It returns
the existing root-free IDs/bytes value, preserves special tokens, and rejects
unknown IDs or capacity overflow. It does not read checkpoint weights or CUDA.

For an installed checkpoint (paths/IDs supplied by the caller):

```mbt nocheck
let root = @approved_fs.ApprovedRoot::open_absolute(model_root)
let tokenizer = @checkpoint_text_input.load_checkpoint_tokenizer(
  root, model_root, "tokenizer.json", tokenizer_label,
  vocabulary_size~, maximum_tokens~,
)
root.close()
let output_bytes = tokenizer.decode_bytes(generated_ids, special_policy=Preserve)
```

Preserve special tokens for diagnostic comparisons. Presentation may deliberately
skip them, but that must not erase an EOS boundary from a correctness result.
An arbitrary partial token vector may end inside a UTF-8 scalar; decoded output
is consequently `Bytes`, not an implicitly repaired string. The black-box loaded
fixture tests encoding, preserved/skipped special tokens, unknown-token rejection
and use after root release.
