///|
/// CUDA instruction binding of the generic single-use forwarding transform.
/// A shared m8n8x4 load supplies two ordered m16n8k16 products. Only the RHS
/// representation is forwarded; reusable Q/probability and F32 accumulators
/// retain their existing owners and numerical representation.
pub struct CudaPrefillFragmentBinding {
  priv program : @attention_physical_ir.OnlineFragmentProgram
  priv score : @physical_tile_ir.FragmentForwarding
  priv value : @physical_tile_ir.FragmentForwarding
  priv query_lifetime : @attention_physical_ir.QueryOperandLifetime
  priv probabilities : @attention_physical_ir.OrderedProbabilityPacking
  priv value_addresses : @physical_tile_ir.OperandRowFactoring
  priv key_read_ownership : @physical_tile_ir.FragmentReadOwnership?
  priv key_operand_reads : @physical_tile_ir.OrderedOperandReads?
} derive(Eq, Debug)

///|
pub extend CudaPrefillFragmentBinding with Eq::{equal, not_equal}

///|
pub extend CudaPrefillFragmentBinding with @debug.Debug::{to_repr}

///|
/// Reject incompatible instruction/layout facts before terminal rendering.
/// Packed BF16 pairs and the row-preserving, eight-element vector layout are
/// required by this CUDA realization, not restrictions on the generic IR.
pub fn CudaPrefillFragmentBinding::new(
  program : @attention_physical_ir.OnlineFragmentProgram,
  subgroup_lanes~ : Int,
  operand_vector_elements~ : Int,
  retained_query_dimensions? : Int = 0,
  joint_query_consumers? : Bool = false,
  right_operand_read_slots? : Int = 0,
) -> CudaPrefillFragmentBinding? {
  guard program.geometry() == (16, 16, 16) &&
    subgroup_lanes == 32 &&
    operand_vector_elements == 8 else {
    return None
  }
  guard program
    .score()
    .forward_right(words=4, instruction_words=2, [0, 2, 1, 3])
    is Some(score) else {
    return None
  }
  guard right_operand_read_slots >= 0 &&
    right_operand_read_slots <= 2 &&
    (
      right_operand_read_slots == 0 ||
      (!joint_query_consumers && retained_query_dimensions == 0)
    ) else {
    return None
  }
  let key_read_ownership = if right_operand_read_slots > 0 {
    score.contiguous_read(provider_group_width=subgroup_lanes / 4)
  } else {
    None
  }
  let key_operand_reads = if right_operand_read_slots > 0 {
    @physical_tile_ir.OrderedOperandReads::new(
      items=score.program().geometry().1,
      slots=right_operand_read_slots,
    )
  } else {
    None
  }
  guard right_operand_read_slots == 0 ||
    (key_read_ownership is Some(_) && key_operand_reads is Some(_)) else {
    return None
  }
  let score = if right_operand_read_slots > 0 {
    program
    .score()
    .forward_right(words=4, instruction_words=2, [0, 1, 2, 3])
    .unwrap()
  } else {
    score
  }
  guard program
    .value()
    .forward_right(words=4, instruction_words=2, [0, 1, 2, 3])
    is Some(value) else {
    return None
  }
  guard @attention_physical_ir.QueryOperandLifetime::new(
      program,
      retained_dimensions=retained_query_dimensions,
      operand_words=4,
      accumulator_words=8,
      joint_right_consumers=joint_query_consumers,
    )
    is Some(query_lifetime) else {
    return None
  }
  guard program.pack_probabilities(
      [0, 0, 1, 1, 0, 0, 1, 1],
      elements_per_word=2,
    )
    is Some(probabilities) else {
    return None
  }
  let (_, _, dimensions) = program.fold().geometry()
  let rotations = if dimensions / 8 % 8 == 0 {
    8
  } else if dimensions / 8 % 4 == 0 {
    4
  } else {
    2
  }
  guard @physical_tile_ir.OperandLayout::new(
      columns=dimensions,
      vector=8,
      rotations~,
      row_group=8 / rotations,
    )
    is Some(layout) else {
    return None
  }
  Some({
    program,
    score,
    value,
    query_lifetime,
    probabilities,
    value_addresses: layout.factor_rows(),
    key_read_ownership,
    key_operand_reads,
  })
}

///|
pub fn CudaPrefillFragmentBinding::key_read_ownership(
  self : CudaPrefillFragmentBinding,
) -> @physical_tile_ir.FragmentReadOwnership? {
  self.key_read_ownership
}

///|
pub fn CudaPrefillFragmentBinding::key_operand_reads(
  self : CudaPrefillFragmentBinding,
) -> @physical_tile_ir.OrderedOperandReads? {
  self.key_operand_reads
}

///|
pub fn CudaPrefillFragmentBinding::probabilities(
  self : CudaPrefillFragmentBinding,
) -> @attention_physical_ir.OrderedProbabilityPacking {
  self.probabilities
}

///|
pub fn CudaPrefillFragmentBinding::value_addresses(
  self : CudaPrefillFragmentBinding,
) -> @physical_tile_ir.OperandRowFactoring {
  self.value_addresses
}

///|
pub fn CudaPrefillFragmentBinding::program(
  self : CudaPrefillFragmentBinding,
) -> @attention_physical_ir.OnlineFragmentProgram {
  self.program
}

///|
pub fn CudaPrefillFragmentBinding::score(
  self : CudaPrefillFragmentBinding,
) -> @physical_tile_ir.FragmentForwarding {
  self.score
}

///|
pub fn CudaPrefillFragmentBinding::value(
  self : CudaPrefillFragmentBinding,
) -> @physical_tile_ir.FragmentForwarding {
  self.value
}

///|
pub fn CudaPrefillFragmentBinding::query_lifetime(
  self : CudaPrefillFragmentBinding,
) -> @attention_physical_ir.QueryOperandLifetime {
  self.query_lifetime
}

///|
pub fn CudaAttentionTileLowering::prefill_fragments(
  self : CudaAttentionTileLowering,
) -> CudaPrefillFragmentBinding? {
  self.prefill_fragments
}

///|
fn cuda_prefill_fragment_binding(
  physical : Result[
    @luna_attention_tile_schedule.AttentionPhysicalProgram,
    @luna_attention_tile_schedule.AttentionTileScheduleRule,
  ],
  retained_query_dimensions : Int,
  joint_query_consumers : Bool,
  right_operand_read_slots : Int,
) -> CudaPrefillFragmentBinding? raise CudaAttentionTileLoweringError {
  guard physical is Ok(QueryFragmentProgram(fold, _)) else { return None }
  guard fold.fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=16,
    )
    is Some(program) &&
    CudaPrefillFragmentBinding::new(
      program,
      subgroup_lanes=32,
      operand_vector_elements=8,
      retained_query_dimensions~,
      joint_query_consumers~,
      right_operand_read_slots~,
    )
    is Some(binding) else {
    raise InvalidCudaAttentionTileLowering(UnsupportedCompute)
  }
  Some(binding)
}
