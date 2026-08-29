///|
/// Admits the current host-owned sampling boundary from an immutable model
/// plan. The terminal producer must be the unique BF16 language-model head
/// whose projection width is the complete vocabulary.
///
/// This function deliberately returns no launch dimensions, operands, source,
/// recipe, entry point, or artifact identity. Sampling is outside the semantic
/// graph and adding such data here would invent an unauthenticated operation.
pub fn admit_host_sampling_boundary(
  model : @plan.ModelPlan,
) -> HostSamplingBoundary raise SamplingBoundaryError {
  let final_output = model.final_output()
  let mut producer : @plan.PlanOperation? = None
  for operation in model.operations() {
    for output in operation.outputs() {
      if output == final_output {
        if producer is Some(_) {
          raise Invalid(FinalOutputProducer)
        }
        producer = Some(operation)
      }
    }
  }
  guard producer is Some(operation) else { raise Invalid(FinalOutputProducer) }
  if operation.kind() != LanguageModelHead {
    raise Unsupported(FinalOutputKind)
  }
  let vocabulary_size = model.shape_constraints().vocabulary_size()
  guard operation.shape() is ProjectionShape(input_width=_, output_width~) &&
    output_width == vocabulary_size else {
    raise Invalid(FinalOutputShape)
  }
  let operation_index = operation.id().as_int()
  let numeric = model.numeric_binding().operations()
  if operation_index < 0 || operation_index >= numeric.length() {
    raise Invalid(FinalOutputDtype)
  }
  if numeric[operation_index].execution().output() !=
    @numeric_contract.ComputeDType::bfloat16() {
    raise Unsupported(FinalOutputDtype)
  }
  if vocabulary_size <= 0 {
    raise Invalid(VocabularySize)
  }
  let logits_row_bytes = Int64::from_int(vocabulary_size) * 2L
  if logits_row_bytes / 2L != Int64::from_int(vocabulary_size) {
    raise SizeOverflow(VocabularySize)
  }
  {
    owner: HostCompletionOwner,
    producer: operation.id(),
    final_output,
    vocabulary_size,
    logits_row_bytes,
  }
}
