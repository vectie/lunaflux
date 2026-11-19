///|
pub(all) enum Component {
  Embedding
  Decoder(Int32)
  FinalNorm
  LanguageHead
  VisionEmbedding
  VisionLayer(Int32)
  VisionFinalNorm
  EarlyVisionMerger
  OutputVisionMerger
} derive(Debug, Eq)

///|
pub extend Component with @debug.Debug::{to_repr}

///|
pub extend Component with Eq::{equal, not_equal}

///|
pub struct Requirement {
  name : String
  shape : ReadOnlyArray[Int64]
  component : Component
}

///|
pub suberror WeightError {
  InvalidWeights(String)
} derive(Debug)

///|
pub extend WeightError with @debug.Debug::{to_repr}

///|
/// Startup binding to already-validated shard metadata. No tensor payload or
/// native handle is retained. file_offset includes the safetensors header.
pub(all) struct Binding {
  name : String
  shape : ReadOnlyArray[Int64]
  component : Component
  shard_index : Int32
  tensor_index : Int32
  file_offset : Int64
  byte_length : Int64
}

///|
pub struct BoundWeights {
  priv bindings : ReadOnlyArray[Binding]
  priv indices : Map[String, Int32]
  priv total_bytes : Int64
}

///|
pub fn BoundWeights::tensor_count(self : BoundWeights) -> Int32 {
  self.bindings.length()
}

///|
pub fn BoundWeights::total_bytes(self : BoundWeights) -> Int64 {
  self.total_bytes
}

///|
pub fn BoundWeights::tensor(self : BoundWeights, index : Int32) -> Binding? {
  self.bindings.get(index)
}

///|
pub fn BoundWeights::find_tensor(
  self : BoundWeights,
  name : String,
) -> Binding? {
  self.indices.get(name).map(i => self.bindings[i])
}
