///|
/// Primary-source startup conversion used only by the official reference
/// checkpoint converter for attention output-A weights. This is schema
/// evidence, not a materializer or device authority.
pub(all) enum DeepSeekV4OutputAConversionContract {
  Fp8E4M3Ue8M0Block128ToBfloat16V1
} derive(Debug, Eq)

///|
pub struct DeepSeekV4OutputAConvertedLayout {
  priv contract : DeepSeekV4OutputAConversionContract
  priv source_weight_shape : ReadOnlyArray[Int64]
  priv source_scale_shape : ReadOnlyArray[Int64]
  priv converted_grouped_shape : ReadOnlyArray[Int64]
}

///|
pub fn output_a_converted_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4OutputAConvertedLayout {
  let groups = spec.output_group_count()
  let rank = spec.o_lora_rank()
  let input = spec.attention_head_count() * spec.head_dimension() / groups
  let rows = groups * rank
  {
    contract: Fp8E4M3Ue8M0Block128ToBfloat16V1,
    source_weight_shape: shape2(rows, input),
    source_scale_shape: scale2(rows, input),
    converted_grouped_shape: [
      groups.to_int64(),
      rank.to_int64(),
      input.to_int64(),
    ],
  }
}

///|
pub fn DeepSeekV4OutputAConvertedLayout::contract(
  self : Self,
) -> DeepSeekV4OutputAConversionContract {
  self.contract
}

///|
pub fn DeepSeekV4OutputAConvertedLayout::source_weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.source_weight_shape
}

///|
pub fn DeepSeekV4OutputAConvertedLayout::source_scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.source_scale_shape
}

///|
pub fn DeepSeekV4OutputAConvertedLayout::converted_grouped_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.converted_grouped_shape
}
