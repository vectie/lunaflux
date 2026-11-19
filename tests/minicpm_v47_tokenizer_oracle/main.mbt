///|
async fn main {
  guard @env.args()[1:] is [assets, oracle, output] else {
    fail("usage: minicpm_v47_tokenizer_oracle PINNED_ASSETS ORACLE NEW_OUTPUT")
  }
  @fs.mkdir(output, recursive=true)
  let model = @model.Spec::parse(
    @fs.read_file(assets + "/config.json").binary()[:],
  )
  let spec = @adapter.prepare_tokenizer(
    model,
    @fs.read_file(assets + "/tokenizer.json").binary()[:],
    @tokenizer.TokenizerDigest::parse("a".repeat(64)),
    {
      max_input_bytes: 16384,
      max_output_tokens: 16384,
      max_decoded_bytes: 65536,
      max_vocab_entries: 250000,
      max_merge_rules: 500000,
      max_token_bytes: 2048,
      max_special_tokens: 128,
    },
  )
  let corpus = corpus(model)
  let corpus_path = output + "/corpus.json"
  @fs.write_file(
    corpus_path,
    corpus.to_json().stringify(),
    create_mode=CreateNew,
  )
  let observed = output + "/upstream.jsonl"
  guard @process.run(oracle, [assets + "/tokenizer.json", corpus_path, observed]) ==
    0 else {
    fail("upstream tokenizer oracle failed")
  }
  let records = @fs.read_file(observed).text().trim().split("\n").to_array()
  guard records.length() == corpus.length() + 1 else {
    fail("incomplete upstream corpus")
  }
  let mut compared_tokens = 0
  for index, text in corpus {
    guard @json.parse(records[index + 1])
      is {
        "text": String(reference_text),
        "ids": Array(reference_ids),
        "decoded": String(decoded),
        ..
      } &&
      reference_text == text else {
      fail("upstream corpus ordering differs")
    }
    let native = spec.encode_text(
      text,
      special_policy=Allow,
      output_limit=16384,
      truncation=RejectOverflow,
    )
    let ids = reference_ids.map(value => {
      guard value is Number(n, ..) && n >= 0 && n < 248144 && n == n.trunc() else {
        fail("invalid upstream token ID")
      }
      n.to_int()
    })
    guard native.token_ids == ids else {
      fail(
        "token IDs differ at corpus case \{index}: text=\{text}, native=\{native.token_ids.to_json().stringify()}, upstream=\{ids.to_json().stringify()}",
      )
    }
    guard spec.decode_text(native.token_ids, special_policy=Preserve) == decoded else {
      fail("decoded text differs at corpus case \{index}")
    }
    compared_tokens += ids.length()
  }
  let report : Json = {
    "passed": true,
    "oracle": @json.parse(records[0]),
    "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
    "declared_asset_revision": @fs.read_file(assets + "/revision.txt")
    .text()
    .trim()
    .to_owned(),
    "cases": corpus.length(),
    "tokens_compared": compared_tokens,
    "scope": "tokenizer.json encode/decode; raw text and LunaFlux-rendered prompts",
    "upstream_chat_template_compared": false,
    "gpu_execution": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(report.stringify(indent=2))
}
