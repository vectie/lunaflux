///|
priv struct DenoiserBindings[T] {
  input : (@weights.InputWeightPacking, T)
  timestep : (@weights.TimestepWeightPacking, T)
  output : (@weights.OutputWeightPacking, T)
  context : (@weights.ContextWeightPacking, T)
  refiners : ReadOnlyArray[(@weights.RefinerWeightPacking, T)]
  layers : ReadOnlyArray[(@weights.BlockWeightPacking, T)]
}

///|
// Resolve immutable packing metadata, independently of device effects. This
// shares the production ordering algorithm with host-only regression fixtures.
fn[T] denoiser_bindings(
  entries : ArrayView[(Packing, T)],
  layer_count : Int,
  refiner_count : Int,
) -> DenoiserBindings[T] raise StartupError {
  guard layer_count > 0 && refiner_count > 0 else { raise InvalidPacking }
  let mut input = None
  let mut timestep = None
  let mut output = None
  let mut context = None
  let layers : Map[Int, (@weights.BlockWeightPacking, T)] = Map([])
  let refiners : Map[Int, (@weights.RefinerWeightPacking, T)] = Map([])
  for (packing, allocation) in entries {
    match packing {
      Input(p) => {
        guard input is None else { raise InvalidPacking }
        input = Some((p, allocation))
      }
      Timestep(p) => {
        guard timestep is None else { raise InvalidPacking }
        timestep = Some((p, allocation))
      }
      Output(p) => {
        guard output is None else { raise InvalidPacking }
        output = Some((p, allocation))
      }
      Context(p) => {
        guard context is None else { raise InvalidPacking }
        context = Some((p, allocation))
      }
      Block(p) => {
        guard p.layer >= 0 && p.layer < layer_count && !layers.contains(p.layer) else {
          raise InvalidPacking
        }
        layers[p.layer] = (p, allocation)
      }
      Refiner(p) => {
        guard p.layer >= 0 &&
          p.layer < refiner_count &&
          !refiners.contains(p.layer) else {
          raise InvalidPacking
        }
        refiners[p.layer] = (p, allocation)
      }
      _ => raise InvalidPacking
    }
  }
  guard input is Some(input) &&
    timestep is Some(timestep) &&
    output is Some(output) &&
    context is Some(context) &&
    layers.length() == layer_count &&
    refiners.length() == refiner_count else {
    raise InvalidPacking
  }
  {
    input,
    timestep,
    output,
    context,
    layers: ReadOnlyArray::from_array(Array::makei(layer_count, i => layers[i])),
    refiners: ReadOnlyArray::from_array(
      Array::makei(refiner_count, i => refiners[i]),
    ),
  }
}

///|
/// Borrow the complete checkpoint-backed prediction and text-refiner weights.
/// Allocation order is irrelevant: retained model packings determine ownership.
/// Keep this lease alive until every consuming queue/program has drained.
/// No payload reads, uploads, hashing or per-step work occurs in this join.
pub fn[T] WeightLease::with_denoiser_weights(
  self : Self,
  component : @weights.MiniMaxH3WeightComponent,
  plan : @joint.JointDiffusionPlan,
  consume : (
    @blocks.PredictionWeights,
    @device.Allocation,
    ReadOnlyArray[(@weights.RefinerWeightPacking, @device.Allocation)],
  ) -> T raise,
) -> T raise {
  guard component == Fl2vaDenoiser || component == Ref2vaDenoiser else {
    raise InvalidPacking
  }
  guard !self.released && !self.owner.failed && !self.owner.closed else {
    raise InvalidState
  }
  let entries : Array[(Packing, @device.Allocation)] = []
  for entry in self.owner.entries {
    if entry.component == component {
      guard entry.owner is Packed(allocation) else { raise InvalidPacking }
      entries.push((entry.packing, allocation))
    }
  }
  let shape = plan.denoiser()
  let bound = denoiser_bindings(
    entries[:],
    shape.layer_count(),
    shape.refiner_layer_count(),
  )
  // TextConditioningProgram consumes the context projection at offset zero.
  guard bound.context.0.weight_offset == 0L else { raise InvalidPacking }
  consume(
    @blocks.PredictionWeights::new(
      bound.timestep,
      bound.input,
      bound.layers[:],
      bound.output,
    ),
    bound.context.1,
    bound.refiners,
  )
}
