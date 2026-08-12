///|
fn tensor_shape(
  host : @materialize.HostTensor,
) -> (Int, Int) raise ReferenceExecutionError {
  let shape = host.tensor().shape()
  match shape.length() {
    1 => {
      let columns = shape[0]
      if columns <= 0L || columns > 2147483647L {
        raise InvalidGraph(ShapeMismatch)
      }
      (1, columns.to_int())
    }
    2 => {
      let rows = shape[0]
      let columns = shape[1]
      if rows <= 0L ||
        columns <= 0L ||
        rows > 2147483647L ||
        columns > 2147483647L {
        raise InvalidGraph(ShapeMismatch)
      }
      (rows.to_int(), columns.to_int())
    }
    _ => raise InvalidGraph(ShapeMismatch)
  }
}

///|
fn load_tensor(
  model : PreparedModel,
  reference : @plan.TensorRef,
) -> @refkernels.Tensor raise ReferenceExecutionError {
  guard model.tensors.get(reference.as_int()) is Some(tensor) else {
    raise InvalidGraph(MissingTensor)
  }
  tensor
}

///|
fn activation_input(
  activations : Array[@refkernels.Activation],
  value : @plan.ValueRef,
) -> @refkernels.Activation raise ReferenceExecutionError {
  guard value is Activation(reference) else {
    raise InvalidGraph(UnsupportedInput)
  }
  guard activations.get(reference.as_int()) is Some(activation) else {
    raise InvalidGraph(MissingActivation)
  }
  activation
}

///|
fn tensor_input(
  model : PreparedModel,
  value : @plan.ValueRef,
) -> @refkernels.Tensor raise ReferenceExecutionError {
  guard value is Tensor(reference) else { raise InvalidGraph(UnsupportedInput) }
  load_tensor(model, reference)
}

///|
/// Decodes and validates immutable BF16 host weights exactly once.
pub fn prepare(
  plan : @plan.ModelPlan,
  weights : @materialize.HostMaterialization,
  limits~ : ExecutionLimits,
) -> PreparedModel raise ReferenceExecutionError {
  if plan.identity() != weights.identity() {
    raise InvalidGraph(IdentityMismatch)
  }
  if weights.length() > limits.max_tensors() {
    raise InvalidGraph(TensorLimitExceeded)
  }
  let tensors : Array[@refkernels.Tensor] = []
  for index in 0..<weights.length() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise InvalidGraph(MissingTensor)
    }
    guard weights.tensor(reference) is Some(host) else {
      raise InvalidGraph(MissingTensor)
    }
    let (rows, columns) = tensor_shape(host)
    let tensor = @refkernels.decode_bfloat16_le(
      host.bytes(),
      rows~,
      columns~,
      limits=limits.kernels(),
    ) catch {
      error => raise Kernel(error)
    }
    tensors.push(tensor)
  }
  { identity: weights.identity(), tensors: ReadOnlyArray::from_array(tensors) }
}
