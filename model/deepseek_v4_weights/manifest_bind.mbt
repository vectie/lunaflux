///|
fn validate_non_overlapping(
  tensors : ArrayView[DeepSeekV4SafetensorsTensor],
) -> Unit raise DeepSeekV4TensorSchemaError {
  let sorted = tensors.to_owned()
  sorted.sort_by((left, right) => {
    let shard_order = left.shard - right.shard
    if shard_order != 0 {
      shard_order
    } else if left.data_start < right.data_start {
      -1
    } else if left.data_start > right.data_start {
      1
    } else {
      0
    }
  })
  for index in 1..<sorted.length() {
    let previous = sorted[index - 1]
    let current = sorted[index]
    if previous.shard == current.shard && previous.data_end > current.data_start {
      raise InvalidByteRange
    }
  }
}

///|
/// Binds an aggregate of already authenticated official safetensors headers to
/// one exact semantic plan. The aggregate may span shards; names must be
/// globally unique and byte ranges may not overlap within a shard.
pub fn bind(
  plan : @deepseek_v4.DeepSeekV4SemanticPlan,
  tensors : ArrayView[DeepSeekV4SafetensorsTensor],
) -> DeepSeekV4WeightBinding raise DeepSeekV4TensorSchemaError {
  let spec = plan.metadata().spec()
  let expected = expected_tensor_count(spec)
  if tensors.length() != expected {
    raise IncorrectTensorCount(expected~, actual=tensors.length())
  }
  validate_non_overlapping(tensors)
  let seen : Map[String, Int] = Map([], capacity=tensors.length())
  let bindings : Array[DeepSeekV4TensorBinding] = []
  for index, tensor in tensors {
    match seen.get(tensor.name) {
      Some(previous) => raise DuplicateTensor(previous, index)
      None => seen.set(tensor.name, index)
    }
    let binding = validate_tensor_semantic(spec, tensor) catch {
      UnknownTensor(_) => raise UnknownTensor(index)
      InvalidTensor(_) => raise InvalidTensor(index)
      error => raise error
    }
    bindings.push(binding)
  }
  let execution = plan.execution_plan()
  if execution.content_digest() != plan.content_digest() {
    raise IdentityMismatch
  }
  let identity = @spec.ModelIdentity::new(
    execution.content_digest(),
    execution.plan_digest(),
  ) catch {
    _ => raise IdentityMismatch
  }
  {
    profile: spec.profile(),
    identity,
    tensors: ReadOnlyArray::from_array(bindings),
  }
}
