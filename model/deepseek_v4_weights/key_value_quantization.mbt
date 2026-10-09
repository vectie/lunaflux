///|
/// Dense pre-normalization key/value storage used by every admitted V4
/// profile. The official inference model declares `wkv` as a replicated
/// `Linear(hidden, head_dimension)` rather than a parallel projection.
pub(all) enum DeepSeekV4KeyValueQuantizationContract {
  Fp8E4M3Ue8M0Block128ReplicatedRowMajorV1
} derive(Debug, Eq)

///|
pub struct DeepSeekV4KeyValueQuantizedLayout {
  priv contract : DeepSeekV4KeyValueQuantizationContract
  priv weight_shape : ReadOnlyArray[Int64]
  priv scale_shape : ReadOnlyArray[Int64]
  priv input_width : Int
  priv output_width : Int
  priv block_width : Int
  priv tensor_parallel_replicated : Bool
}

///|
pub fn key_value_quantized_layout(
  spec : @deepseek_v4_spec.DeepSeekV4ModelSpec,
) -> DeepSeekV4KeyValueQuantizedLayout {
  let input_width = spec.hidden_size()
  let output_width = spec.kv_head_count() * spec.head_dimension()
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
pub fn DeepSeekV4KeyValueQuantizedLayout::contract(
  self : Self,
) -> DeepSeekV4KeyValueQuantizationContract {
  self.contract
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::weight_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.weight_shape
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::scale_shape(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.scale_shape
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::input_width(self : Self) -> Int {
  self.input_width
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::output_width(self : Self) -> Int {
  self.output_width
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::block_width(self : Self) -> Int {
  self.block_width
}

///|
pub fn DeepSeekV4KeyValueQuantizedLayout::tensor_parallel_replicated(
  self : Self,
) -> Bool {
  self.tensor_parallel_replicated
}
