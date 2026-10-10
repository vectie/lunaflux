///|
/// (exclusive launch end, workspace region, tensor name), derived from the
/// binding recipe. Combined Q/K operators have two observable outputs.
fn first_layer_operator_outputs() -> ReadOnlyArray[(Int, Int, String)] {
  let outputs : Array[(Int, Int, String)] = [(1, 0, "embedding")]
  for ordinal, function in layer_function_order() {
    let end = ordinal + 2
    match function {
      1 =>
        outputs.push(
          (end, 1, if ordinal == 0 { "norm-input" } else { "norm-mlp" }),
        )
      2 => outputs.push((end, 2, "q"))
      3 =>
        outputs.push(
          (
            end,
            if ordinal == 2 {
              3
            } else {
              4
            },
            if ordinal == 2 {
              "k"
            } else {
              "v"
            },
          ),
        )
      4 => {
        outputs.push((end, 2, "q-norm"))
        outputs.push((end, 3, "k-norm"))
      }
      5 => {
        outputs.push((end, 2, "q-rotary"))
        outputs.push((end, 3, "k-rotary"))
      }
      6 => outputs.push((end, 5, "attention"))
      7 => outputs.push((end, 6, "output"))
      8 => outputs.push((end, 7, "activation"))
      9 => outputs.push((end, 6, "down"))
      10 =>
        outputs.push(
          (end, 0, if ordinal == 8 { "residual-attn" } else { "residual-mlp" }),
        )
      _ => abort("unsupported text encoder operator recipe")
    }
  }
  ReadOnlyArray::from_array(outputs)
}

///|
fn first_layer_operator_boundaries() -> ReadOnlyArray[(Int, Int)] {
  ReadOnlyArray::from_array(
    Array::makei(layer_function_order().length() + 1, i => (i, i + 1)),
  )
}

///|
/// Exact host snapshot storage for the explicit first-layer diagnostic. Pure
/// planning only; no tensor allocation, filesystem work or payload hashing.
pub fn TextEncoderSource::first_layer_observation_bytes(self : Self) -> Int64 {
  let regions = self.plan.workspace_regions()
  for (_, region, _) in first_layer_operator_outputs(); bytes = 0L {
    continue bytes + regions[region].1
  } nobreak {
    bytes
  }
}

///|
/// Synchronized operator observations are opt-in and never used by prepare().
/// Failed execution/callback cannot submit subsequent kernels or publish a
/// complete encoder output. Existing close/cancel own every partial queue.
pub fn EncoderRequest::execute_operator_observed(
  self : Self,
  on_tensor~ : (String, Bytes) -> Unit raise,
) -> Unit raise {
  guard self.phase == Prepared &&
    self.queue is None &&
    self.observations.length() == first_layer_operator_boundaries().length() &&
    !self.allocations.is_empty() else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  let regions = self.source.plan.workspace_regions()
  observe_operator_outputs(
    self.observations.length(),
    step => {
      let (queue, _) = self.observations[step]
      queue.enqueue(0)
      queue.record_completion()
      queue.wait_completion()
    },
    region => {
      let (offset, bytes) = regions[region]
      self.allocations[0].copy_to_host(offset, bytes)
    },
    on_tensor,
  )
  self.phase = PrefixComplete
}

///|
fn observe_operator_outputs(
  count : Int,
  run : (Int) -> Unit raise,
  snapshot : (Int) -> Bytes raise,
  consume : (String, Bytes) -> Unit raise,
) -> Unit raise {
  let outputs = first_layer_operator_outputs()
  for step in 0..<count {
    run(step)
    for (end, region, name) in outputs {
      if end == step + 1 {
        consume(name, snapshot(region))
      }
    }
  }
}
