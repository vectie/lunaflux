///|
fn region(value : @layout.Region) -> Json {
  { "address": value.address, "bytes": value.bytes }
}

///|
/// Inert startup planner. No device is opened and no checkpoint is hashed.
async fn main {
  guard @env.args()[1:] is [contract_path, config_path, label] else {
    fail("lunachip_prepare CONTRACT_JSON QWEN_CONFIG_JSON CONTENT_LABEL_64_HEX")
  }
  guard @fs.file_size(contract_path) <= 4096L &&
    @fs.file_size(config_path) <= 65536L else {
    fail("oversized configuration")
  }
  let device = @lunachip.parse_contract(@fs.read_file(contract_path).text())
  let metadata = @config_reader.parse_qwen3_string(
    @fs.read_file(config_path).text(),
    @spec.ContentDigest::from_sha256(label),
    max_bytes=65536,
    max_depth=16,
  )
  let plan = @layout.prepare(device, metadata)
  let weights : Array[Json] = []
  for ordinal in 0..<plan.weight_count() {
    weights.push({
      "ordinal": ordinal,
      "bf16": region(plan.weight(ordinal).unwrap()),
      "expanded_norm_fp32": match plan.expanded_norm(ordinal) {
        Some(value) => region(value)
        None => Json::null()
      },
    })
  }
  let kv : Array[Json] = []
  for layer in 0..<28 {
    for value in [false, true] {
      let first = plan.kv_head(layer, value, 0, 0)
      let last = plan.kv_head(layer, value, 2047, 7)
      kv.push({
        "layer": layer,
        "value": value,
        "address": first.address,
        "bytes": last.address + last.bytes - first.address,
      })
    }
  }
  let result : Json = {
    "schema": "lunaflux.lunachip-qwen3-preparation/v1",
    "content_label": label,
    "identity_kind": "supplied-association-label",
    "numeric_profile": "qwen-layer-ordered-fp32-approx-v1",
    "batch": 1,
    "context": plan.context(),
    "kv_dtype": "FP32",
    "weights": weights,
    "weight_bytes": plan.weight_bytes(),
    "lm_head_alias_ordinal": 0,
    "kv": kv,
    "kv_bytes": plan.kv_bytes(),
    "workspace": region(plan.workspace()),
    "reserve": region(plan.reserve()),
    "free_bytes": plan.free_bytes(),
    "full_model_lowered": false,
    "device_opened": false,
    "scope": "Complete model placement only; full-model lowering and numerical/device qualification remain open.",
  }
  println(result.stringify(indent=2))
}
