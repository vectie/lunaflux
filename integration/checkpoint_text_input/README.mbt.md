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
