///|
pub struct DeepSeekV4PackedExpertWeight {
  priv stem : String
  priv representation : @precision_format.Representation
}

///|
pub fn DeepSeekV4PackedExpertWeight::payload_name(self : Self) -> String {
  self.stem + ".weight"
}

///|
pub fn DeepSeekV4PackedExpertWeight::scales_name(self : Self) -> String {
  self.stem + ".scale"
}

///|
pub fn DeepSeekV4PackedExpertWeight::representation(
  self : Self,
) -> @precision_format.Representation {
  self.representation
}

///|
/// Family-only names and geometry. FP4 experts retain per-row block32 scales;
/// FP8 experts retain the checkpoint's 128x128 scale grid. No device details.
pub fn packed_expert_weights(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
) -> Array[DeepSeekV4PackedExpertWeight] raise {
  guard layer_index >= 0 &&
    layer_index < spec.layer_count() &&
    !expert_indices.is_empty() else {
    fail("invalid routed expert layer or empty partition")
  }
  let seen : Set[Int] = Set([])
  let weights = []
  for expert in expert_indices {
    guard expert >= 0 &&
      expert < spec.routed_expert_count() &&
      !seen.contains(expert) else {
      fail("invalid or duplicate routed expert")
    }
    seen.add(expert)
    for
      (name, rows, columns) in [
        ("w1", spec.moe_intermediate_size(), spec.hidden_size()),
        ("w3", spec.moe_intermediate_size(), spec.hidden_size()),
        ("w2", spec.hidden_size(), spec.moe_intermediate_size()),
      ] {
      let representation = match spec.expert_storage() {
        Fp4E2M1 =>
          @precision_format.Representation::new(format=MxFp4, rows~, columns~)
        Fp8E4M3 =>
          @precision_format.Representation::new(
            format=MxFp8E4M3,
            rows~,
            columns~,
            grouping=Block(rows=128, columns=128),
          )
      }
      weights.push({
        stem: "layers.\{layer_index}.ffn.experts.\{expert}.\{name}",
        representation,
      })
    }
  }
  weights
}
