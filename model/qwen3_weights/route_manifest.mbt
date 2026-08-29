///|
fn append_qwen3_route_name(
  output : StringBuilder,
  ordinal : Int,
  name : StringView,
) -> Unit {
  output.write_string(
    "tensor=" + ordinal.to_string() + "|" + name.to_owned() + "\n",
  )
}

///|
fn qwen3_route_manifest_sha256(
  plan : @plan.ModelPlan,
  inspection : @numeric_weight_file.NumericWeightFileInspection,
) -> String {
  let encoded = StringBuilder()
  encoded.write_string(
    "schema=lunaflux.qwen3-route-manifest.v1\ncontent=" +
    plan.identity().content().as_hex() +
    "\nplan=" +
    plan.identity().plan().as_hex() +
    "\nnumeric_schema=" +
    inspection.numeric_schema_digest().as_hex() +
    "\nartifact=" +
    inspection.artifact_digest().as_hex() +
    "\ntied_lm_head_tensor_ref=0\n",
  )
  let mut ordinal = 0
  append_qwen3_route_name(encoded, ordinal, "model.embed_tokens.weight")
  ordinal += 1
  let layer_count = (plan.tensor_count() - 2) / 11
  for layer in 0..<layer_count {
    let prefix = "model.layers.\{layer}"
    for
      suffix in [
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "self_attn.o_proj.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "mlp.down_proj.weight",
      ] {
      append_qwen3_route_name(encoded, ordinal, prefix + "." + suffix)
      ordinal += 1
    }
  }
  append_qwen3_route_name(encoded, ordinal, "model.norm.weight")
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(encoded.to_string())))
}
