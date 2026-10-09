///|
/// Dense Query-A storage used by every admitted V4 profile. Unlike Query-B,
/// `inference/convert.py` does not shard this weight for tensor parallelism.
pub(all) enum DeepSeekV4QueryAQuantizationContract {
  Fp8E4M3Ue8M0Block128ReplicatedRowMajorV1
} derive(Debug, Eq)

///|
pub struct DeepSeekV4QueryAQuantizedLayout {
  priv contract : DeepSeekV4QueryAQuantizationContract
  priv weight_shape : ReadOnlyArray[Int64]
  priv scale_shape : ReadOnlyArray[Int64]
  priv input_width : Int
  priv output_width : Int
  priv block_width : Int
  priv tensor_parallel_replicated : Bool
}

///|
pub fn query_a_quantized_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4QueryAQuantizedLayout {
  let input_width = spec.hidden_size()
  let output_width = spec.q_lora_rank()
  {
    contract: Fp8E4M3Ue8M0Block128ReplicatedRowMajorV1,
    weight_shape: shape2(output_width, input_width),
    scale_shape: scale2(output_width, input_width),
    input_width,
    output_width,
    block_width: 128,
    tensor_parallel_replicated: true,
  }
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::contract(
  self : Self,
) -> DeepSeekV4QueryAQuantizationContract {
  self.contract
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weight_shape
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.scale_shape
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4QueryAQuantizedLayout::tensor_parallel_replicated(
  self : Self,
) -> Bool {
  self.tensor_parallel_replicated
}
