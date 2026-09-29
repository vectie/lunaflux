///|
/// CUDA instruction binding of the generic single-use forwarding transform.
/// A shared m8n8x4 load supplies two ordered m16n8k16 products. Only the RHS
/// representation is forwarded; reusable Q/probability and F32 accumulators
/// retain their existing owners and numerical representation.
pub struct CudaPrefillFragmentBinding {
  priv program : @attention_physical_ir.OnlineFragmentProgram
  priv score : @physical_tile_ir.FragmentForwarding
  priv value : @physical_tile_ir.FragmentForwarding
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
  guard program
    .value()
    .forward_right(words=4, instruction_words=2, [0, 1, 2, 3])
    is Some(value) else {
    return None
  }
  Some({ program, score, value, })
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
    )
    is Some(binding) else {
    raise InvalidCudaAttentionTileLowering(UnsupportedCompute)
  }
  Some(binding)
}
