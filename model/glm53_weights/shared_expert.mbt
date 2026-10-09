///|
/// Shared experts in the installed Flash NVFP4 checkpoint remain dense BF16.
/// They are replicated, not members of a routed expert partition.
pub struct Glm53SharedExpertWeight {
  priv layer : Int
  priv name : String
  priv representation : @precision_format.Representation
}

///|
pub fn Glm53SharedExpertWeight::layer_index(self : Self) -> Int {
  self.layer
}

///|
pub fn Glm53SharedExpertWeight::payload_name(self : Self) -> String {
  self.name
}

///|
pub fn Glm53SharedExpertWeight::representation(
  self : Self,
) -> @precision_format.Representation {
  self.representation
}

///|
/// Returns the gate/up/down matrices without scale planes or BF16 expansion.
pub fn shared_expert_weights(
  layer_index~ : Int,
) -> Array[Glm53SharedExpertWeight] raise Glm53NvFp4ExpertError {
  guard layer_index >= 3 && layer_index <= 45 else {
    raise InvalidMoeLayer(layer_index)
  }
  [
    ("gate_proj", 2048, 4096),
    ("up_proj", 2048, 4096),
    ("down_proj", 4096, 2048),
  ].map(shape => {
    let (stem, rows, columns) = shape
    {
      layer: layer_index,
      name: "model.language_model.layers.\{layer_index}.mlp.shared_experts.\{stem}.weight",
      representation: @precision_format.Representation::new(
        format=Bf16,
        rows~,
        columns~,
      ) catch {
        error => raise Precision(error)
      },
    }
  })
}
