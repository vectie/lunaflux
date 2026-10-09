///|
/// Official checkpoint contract retained by `inference/convert.py` for the
/// second attention output projection. The converter shards axis 1 for tensor
/// parallelism and does not dequantize this pair.
pub(all) enum DeepSeekV4OutputBQuantizationContract {
  Fp8E4M3Ue8M0Block128RowMajorV1
} derive(Debug, Eq)

///|
pub struct DeepSeekV4OutputBQuantizedLayout {
  priv contract : DeepSeekV4OutputBQuantizationContract
  priv weight_shape : ReadOnlyArray[Int64]
  priv scale_shape : ReadOnlyArray[Int64]
  priv input_width : Int
  priv output_width : Int
  priv block_width : Int
  priv tensor_parallel_shard_axis : Int
}

///|
pub fn output_b_quantized_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4OutputBQuantizedLayout {
  let input_width = spec.output_group_count() * spec.o_lora_rank()
  let output_width = spec.hidden_size()
  {
    contract: Fp8E4M3Ue8M0Block128RowMajorV1,
    weight_shape: shape2(output_width, input_width),
    scale_shape: scale2(output_width, input_width),
    input_width,
    output_width,
    block_width: 128,
    tensor_parallel_shard_axis: 1,
  }
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::contract(
  self : Self,
) -> DeepSeekV4OutputBQuantizationContract {
  self.contract
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weight_shape
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.scale_shape
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4OutputBQuantizedLayout::tensor_parallel_shard_axis(
  self : Self,
) -> Int {
  self.tensor_parallel_shard_axis
}
