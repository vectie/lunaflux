///|
/// Model-owned checkpoint and routing identity. Numeric execution plans do not
/// contain checkpoint namespaces, and base APIs never reinterpret an out-of-
/// range base layer as a prediction stage.
pub(all) enum DeepSeekV4BlockAddress {
  BaseLayer(Int)
  PredictionStage(Int)
} derive(Debug, Eq)

///|
pub extend DeepSeekV4BlockAddress with Eq::{equal, not_equal}

///|
pub extend DeepSeekV4BlockAddress with @debug.Debug::{to_repr}

///|
pub fn DeepSeekV4BlockAddress::execution_layer(
  self : Self,
  spec : DeepSeekV4ModelSpec,
) -> Int raise DeepSeekV4SpecError {
  match self {
    BaseLayer(index) if index >= 0 && index < spec.layer_count() => index
    PredictionStage(index) if spec.has_dspark() &&
      index >= 0 &&
      index < spec.prediction_stage_count() => spec.layer_count() + index
    _ => raise UnsupportedProfile
  }
}

///|
pub fn DeepSeekV4BlockAddress::checkpoint_prefix(
  self : Self,
  spec : DeepSeekV4ModelSpec,
) -> String raise DeepSeekV4SpecError {
  ignore(self.execution_layer(spec))
  match self {
    BaseLayer(index) => "layers.\{index}."
    PredictionStage(index) => "mtp.\{index}."
  }
}

///|
pub fn DeepSeekV4BlockAddress::symbol_prefix(
  self : Self,
  spec : DeepSeekV4ModelSpec,
) -> String raise DeepSeekV4SpecError {
  ignore(self.execution_layer(spec))
  match self {
    BaseLayer(index) => "deepseek_l\{index}"
    PredictionStage(index) => "deepseek_prediction_s\{index}"
  }
}

///|
pub fn DeepSeekV4BlockAddress::uses_token_hash(
  self : Self,
  spec : DeepSeekV4ModelSpec,
) -> Bool raise DeepSeekV4SpecError {
  self.execution_layer(spec) < spec.hash_layer_count()
}
