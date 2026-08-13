///|
/// SHA-256 identity of immutable model artifact content.
pub struct ContentDigest {
  hex : String
} derive(Debug, Eq)

///|
/// SHA-256 identity of a canonical immutable model plan.
pub struct PlanDigest {
  hex : String
} derive(Debug, Eq)

///|
/// Pair of identities that salts model- and plan-specific runtime state.
pub struct ModelIdentity {
  content : ContentDigest
  plan : PlanDigest
} derive(Debug, Eq)

///|
fn is_lower_sha256(hex : StringView) -> Bool {
  hex.length() == 64 &&
  hex.all(c => c.is_ascii_digit() || ('a' <= c && c <= 'f'))
}

///|
/// Validates a lowercase hexadecimal SHA-256 content identity.
pub fn ContentDigest::from_sha256(
  hex : String,
) -> ContentDigest raise ModelSpecError {
  if !is_lower_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn ContentDigest::as_hex(self : ContentDigest) -> String {
  self.hex
}

///|
/// Validates a lowercase hexadecimal SHA-256 plan identity.
pub fn PlanDigest::from_sha256(hex : String) -> PlanDigest raise ModelSpecError {
  if !is_lower_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn PlanDigest::as_hex(self : PlanDigest) -> String {
  self.hex
}

///|
fn append_u64_le(output : Array[Byte], value : UInt64) -> Unit {
  for byte in value.to_le_bytes() {
    output.push(byte)
  }
}

///|
fn append_int(output : Array[Byte], value : Int) -> Unit {
  append_u64_le(output, Int64::from_int(value).reinterpret_as_uint64())
}

///|
fn append_dtype(output : Array[Byte], dtype : ModelDType) -> Unit {
  output.push(
    match dtype {
      BFloat16 => 1
      Float16 => 2
      Float32 => 3
    },
  )
}

///|
/// Derives the plan identity from validated semantic model data.
///
/// The fixed prefix is the dense-Llama plan schema version. Any builder change
/// that alters operation semantics must introduce a new prefix before release.
/// Integers and IEEE-754 bit patterns use fixed little-endian encodings, so the
/// digest is independent of locale and number formatting.
fn LlamaModelSpec::plan_digest_with_schema(
  self : LlamaModelSpec,
  schema : Bytes,
) -> PlanDigest {
  let encoded : Array[Byte] = []
  for byte in schema {
    encoded.push(byte)
  }
  append_int(encoded, self.vocabulary_size())
  append_int(encoded, self.hidden_size())
  append_int(encoded, self.intermediate_size())
  append_int(encoded, self.layer_count())
  append_int(encoded, self.attention_head_count())
  append_int(encoded, self.kv_head_count())
  append_int(encoded, self.head_dimension())
  append_int(encoded, self.context_length())
  append_u64_le(encoded, self.rope_theta().reinterpret_as_uint64())
  append_u64_le(encoded, self.rms_norm_epsilon().reinterpret_as_uint64())
  append_dtype(encoded, self.weight_dtype())
  append_dtype(encoded, self.kv_dtype())
  {
    hex: @crypto.bytes_to_hex_string(@crypto.sha256(Bytes::from_array(encoded))),
  }
}

///|
/// Derives the full-context plan identity with explicit semantic KV layer
/// ownership. The v2 prefix distinguishes this graph contract from the older
/// operation-order-implied schema.
pub fn LlamaModelSpec::canonical_plan_digest(
  self : LlamaModelSpec,
) -> PlanDigest {
  self.plan_digest_with_schema(b"lunaflux.dense-llama-plan.v2.full-context\x00")
}

///|
/// Derives the distinct paged-KV semantic plan identity. Merely sharing model
/// weights and geometry does not make full-context and stateful graphs the
/// same immutable plan.
pub fn LlamaModelSpec::canonical_paged_plan_digest(
  self : LlamaModelSpec,
) -> PlanDigest {
  self.plan_digest_with_schema(b"lunaflux.dense-llama-plan.v2.paged-kv-v1\x00")
}

///|
/// Constructs an immutable model identity from independently verified digests.
pub fn ModelIdentity::new(
  content : ContentDigest,
  plan : PlanDigest,
) -> ModelIdentity {
  { content, plan }
}

///|
pub fn ModelIdentity::content(self : ModelIdentity) -> ContentDigest {
  self.content
}

///|
pub fn ModelIdentity::plan(self : ModelIdentity) -> PlanDigest {
  self.plan
}
