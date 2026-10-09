///|
pub(all) suberror Glm53NvFp4ExpertError {
  InvalidMoeLayer(Int)
  InvalidExpert(Int)
  DuplicateExpert(Int)
  EmptyExpertPartition
  Precision(@precision_format.PrecisionError)
} derive(Debug, Eq)

///|
pub extend Glm53NvFp4ExpertError with Eq::{equal, not_equal}

///|
pub extend Glm53NvFp4ExpertError with @debug.Debug::{to_repr}

///|
/// Pure checkpoint mapping for one projection. The logical GLM graph is
/// unchanged; ModelOpt's three physical planes implement its weight operand.
pub struct Glm53NvFp4ExpertWeight {
  priv layer : Int
  priv expert : Int
  priv payload_name : String
  priv representation : @precision_format.Representation
}

///|
pub fn Glm53NvFp4ExpertWeight::layer_index(self : Self) -> Int {
  self.layer
}

///|
pub fn Glm53NvFp4ExpertWeight::expert_index(self : Self) -> Int {
  self.expert
}

///|
pub fn Glm53NvFp4ExpertWeight::payload_name(self : Self) -> String {
  self.payload_name
}

///|
pub fn Glm53NvFp4ExpertWeight::scales_name(self : Self) -> String {
  self.payload_name + "_scale"
}

///|
pub fn Glm53NvFp4ExpertWeight::global_scale_name(self : Self) -> String {
  self.payload_name + "_scale_2"
}

///|
pub fn Glm53NvFp4ExpertWeight::representation(
  self : Self,
) -> @precision_format.Representation {
  self.representation
}

///|
/// Expert partitions need not be contiguous: placement is supplied by the
/// caller. Output is expert-major gate/up/down, with no activation quantization
/// implied and no full-bank BF16 expansion. Layer 45 is the checkpoint's MTP
/// layer; layers 0..2 are dense and therefore not routed expert banks.
pub fn nvfp4_expert_weights(
  layer_index~ : Int,
  expert_indices~ : ArrayView[Int],
) -> Array[Glm53NvFp4ExpertWeight] raise Glm53NvFp4ExpertError {
  if layer_index < 3 || layer_index > 45 {
    raise InvalidMoeLayer(layer_index)
  }
  if expert_indices.is_empty() {
    raise EmptyExpertPartition
  }
  let seen : Set[Int] = Set([])
  let weights = []
  for expert in expert_indices {
    if expert < 0 || expert >= 288 {
      raise InvalidExpert(expert)
    }
    if seen.contains(expert) {
      raise DuplicateExpert(expert)
    }
    seen.add(expert)
    let prefix = "model.language_model.layers.\{layer_index}.mlp.experts.\{expert}."
    for
      (stem, rows, columns) in [
        ("gate_proj", 2048, 4096),
        ("up_proj", 2048, 4096),
        ("down_proj", 4096, 2048),
      ] {
      weights.push({
        layer: layer_index,
        expert,
        payload_name: prefix + stem + ".weight",
        representation: @precision_format.Representation::new(
          format=NvFp4,
          rows~,
          columns~,
        ) catch {
          error => raise Precision(error)
        },
      })
    }
  }
  weights
}
