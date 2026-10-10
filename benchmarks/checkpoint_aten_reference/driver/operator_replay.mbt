///|
/// Offline numerical attribution only. Retain each supplied native intermediate
/// under the same explicit session budget, then substitute identical operands
/// into independent ATen operations. Never imported by production LunaFlux.
priv struct OperatorReplayInputs {
  tensors : Map[String, @native.Tensor]
}

///|
fn OperatorReplayInputs::load(
  session : @native.Session,
  root : String,
  rows : Int,
) -> Self raise {
  let owner = { tensors: Map([]), }
  errdefer owner.close()
  for
    (name, width) in [
      ("embedding", 5120),
      ("norm-input", 5120),
      ("q", 8192),
      ("k", 1024),
      ("v", 1024),
      ("q-norm", 8192),
      ("k-norm", 1024),
      ("q-rotary", 8192),
      ("k-rotary", 1024),
      ("attention", 8192),
      ("output", 5120),
      ("residual-attn", 5120),
      ("norm-mlp", 5120),
      ("activation", 25600),
      ("down", 5120),
    ] {
    owner.tensors[name] = session.load(
      root + "/" + name + ".bf16",
      0L,
      rows,
      width,
    )
  }
  owner
}

///|
fn OperatorReplayInputs::close(self : Self) -> Unit {
  for _, tensor in self.tensors {
    tensor.close()
  }
  self.tensors.clear()
}

///|
fn replay_operand(
  replay : OperatorReplayInputs?,
  name : String,
  ordinary : @native.Tensor,
) -> @native.Tensor {
  match replay {
    Some(inputs) => inputs.tensors[name]
    None => ordinary
  }
}
