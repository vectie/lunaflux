///|
const PINNED_CONFIG_SHA256 : String = "9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622"

///|
const PINNED_MODEL_SHA256 : String = "852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30"

///|
const PINNED_TOKENIZER_SHA256 : String = "fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b"

///|
const PINNED_CORPUS_SHA256 : String = "f4ade303b1b7b3264e86caa15b77cdb2805244e1142b9b74225ed3f36438f1d0"

///|
const PINNED_REVISION : String = "3579d71fd57e04f5a364d824d3a2ec3e913dbb67"

///|
suberror ApprovedBf16PhysicalError {
  InvalidInput(String)
  InvalidCompiledSet(String)
  NumericalMismatch(String)
} derive(Debug)

///|
struct ApprovedBf16Inputs {
  metadata : @spec.LlamaModelMetadata
  plan : @plan.ModelPlan
  bindings : @llama_weights.LlamaWeightBindings
  corpus : Json
}

///|
fn approved_sha256(bytes : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn approved_header_limits() -> @safetensors.HeaderLimits raise {
  @safetensors.HeaderLimits::new(
    max_header_bytes=4096,
    max_tensor_count=21,
    max_rank=2,
    max_name_length=128,
    max_payload_bytes=208544L,
    max_materialized_tensor_bytes=208544L,
  )
}

///|
fn approved_tokenizer_limits() -> @tokenizer.TokenizerLimits {
  {
    max_input_bytes: 4096,
    max_output_tokens: 4096,
    max_decoded_bytes: 65536,
    max_vocab_entries: 3000,
    max_merge_rules: 64,
    max_token_bytes: 64,
    max_special_tokens: 3,
  }
}

///|
async fn read_approved_file(root : String, name : String) -> Bytes {
  @fs.read_file("\{root}/\{name}").binary()
}

///|
async fn load_approved_bf16_inputs(root : String) -> ApprovedBf16Inputs {
  let canonical = @fs.realpath(root)
  if canonical != root {
    raise InvalidInput("model root must be canonical and absolute")
  }
  let config = read_approved_file(root, "config.json")
  let model = read_approved_file(root, "model.safetensors")
  let tokenizer_bytes = read_approved_file(root, "upstream_tokenizer.json")
  let corpus_bytes = read_approved_file(root, "corpus.json")
  if approved_sha256(config) != PINNED_CONFIG_SHA256 ||
    approved_sha256(model) != PINNED_MODEL_SHA256 ||
    approved_sha256(tokenizer_bytes) != PINNED_TOKENIZER_SHA256 ||
    approved_sha256(corpus_bytes) != PINNED_CORPUS_SHA256 {
    raise InvalidInput("approved model input digest mismatch")
  }
  let content = @spec.ContentDigest::from_sha256(PINNED_MODEL_SHA256)
  let metadata = @config_reader.parse_bytes(
    config,
    content,
    max_bytes=2048,
    max_depth=8,
  )
  let tensors = @safetensors_reader.read_metadata(
    model,
    limits=approved_header_limits(),
  )
  let plan = @llama.build_paged(
    metadata,
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let bindings = @llama_weights.bind(metadata, plan, tensors)
  let tokenizer = @json_reader.parse_bytes(
    tokenizer_bytes,
    digest=@tokenizer.TokenizerDigest::parse(PINNED_TOKENIZER_SHA256),
    limits=approved_tokenizer_limits(),
    max_bytes=65536,
    max_depth=32,
  )
  let encoded = tokenizer.encode_text(
    "",
    special_policy=AsOrdinary,
    output_limit=8,
    truncation=RejectOverflow,
  )
  if encoded.token_ids != [1] || encoded.truncated {
    raise InvalidInput("approved tokenizer empty-input identity mismatch")
  }
  let corpus = @json.parse(@utf8.decode(corpus_bytes))
  guard corpus is Object(fields) &&
    fields["revision"] is String(PINNED_REVISION) else {
    raise InvalidInput("reference corpus revision mismatch")
  }
  { metadata, plan, bindings, corpus }
}
