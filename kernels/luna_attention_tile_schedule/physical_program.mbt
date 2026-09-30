///|
/// One authoritative terminal program, selected before device source exists.
/// Numeric laws remain distinct: scalar F32 probabilities are not silently
/// substituted for the BF16 matrix probability fold.
pub(all) enum AttentionPhysicalProgram {
  DirectKeyProgram(@attention_physical_ir.KeyFold)
  GroupedKeyProgram(@attention_physical_ir.KeyFold, AttentionGroupedEffectPlan)
  SplitKeyProgram(@attention_physical_ir.KeyFold, AttentionGroupedEffectPlan)
  BlockwiseKeyProgram(
    @attention_physical_ir.BlockwiseFold,
    AttentionGroupedEffectPlan
  )
  SharedScoreProgram(AttentionSharedScoreProgram)
  QueryFragmentProgram(
    @attention_physical_ir.OnlineFold,
    AttentionQueryEffectPlan
  )
} derive(Debug, Eq)

///|
pub extend AttentionPhysicalProgram with Eq::{equal, not_equal}

///|
pub extend AttentionPhysicalProgram with @debug.Debug::{to_repr}

///|
pub struct AttentionSharedScoreProgram {
  priv transfer : AttentionScoreTransferMode
  priv resident_statistics : Bool
  priv resident_output : Bool
  priv matrix_probability : Bool
  priv factor_shared_key : Bool
  priv query_offsets : ReadOnlyArray[Int]
} derive(Debug, Eq)

///|
pub extend AttentionSharedScoreProgram with Eq::{equal, not_equal}

///|
pub extend AttentionSharedScoreProgram with @debug.Debug::{to_repr}

///|
/// Transport and matrix fragment capabilities are explicit backend inputs;
/// no target/vendor name participates in this pure refinement.
pub fn AttentionTileSchedule::physical_program(
  self : AttentionTileSchedule,
  supports_immediate_copy~ : Bool,
  fragment_rows~ : Int,
  factor_shared_key~ : Bool,
) -> AttentionPhysicalProgram raise AttentionTileScheduleError {
  guard fragment_rows > 0 else {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  let queries = self.loops[0].tile
  let dimensions = self.loops[self.loops.length() - 1].extent
  guard @attention_physical_ir.KeyFold::new(dimensions~) is Some(key_fold) else {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  match self.compute {
    AttentionGroupedQueryBlockwiseDot
    | AttentionGroupedQueryDualScoreBlockwiseDot
    | AttentionGroupedQueryFmaBlockwiseDot => {
      let keys = if self.key_value_partitions == 1 {
        self.loops[2].tile
      } else {
        self.loops[3].tile
      }
      let score_owners = if self.compute ==
        AttentionGroupedQueryDualScoreBlockwiseDot {
        2
      } else {
        1
      }
      guard @attention_physical_ir.BlockwiseFold::new(
          dimensions~,
          keys~,
          score_owners~,
          contraction=self.compute == AttentionGroupedQueryFmaBlockwiseDot,
        )
        is Some(fold) else {
        raise InvalidAttentionTileSchedule(InvalidTile)
      }
      BlockwiseKeyProgram(
        fold,
        AttentionGroupedEffectPlan::new(
          self.memory,
          stages=self.pipeline_stages,
          readiness=self.transfer_readiness(),
        ),
      )
    }
    AttentionScalarDot => {
      guard self.memory == AttentionDirectGlobal else {
        raise InvalidAttentionTileSchedule(InvalidLoopNest)
      }
      DirectKeyProgram(key_fold)
    }
    AttentionGroupedQuerySubgroupDot | AttentionGroupedQuerySplitSubgroupDot => {
      let effects = AttentionGroupedEffectPlan::new(
        self.memory,
        stages=self.pipeline_stages,
        readiness=self.transfer_readiness(),
      )
      if self.compute == AttentionGroupedQuerySplitSubgroupDot {
        SplitKeyProgram(key_fold, effects)
      } else {
        GroupedKeyProgram(key_fold, effects)
      }
    }
    AttentionMatrixDot | AttentionMatrixQkPv =>
      if self.fragment_ownership() == QueryOwnedFragments {
        guard self.physical_online_fold() is Some(fold) else {
          raise InvalidAttentionTileSchedule(InvalidTile)
        }
        QueryFragmentProgram(
          fold,
          AttentionQueryEffectPlan::new(
            self.memory,
            stages=self.pipeline_stages,
            readiness=self.transfer_readiness(),
          ),
        )
      } else {
        let matrix_probability = self.compute == AttentionMatrixQkPv
        SharedScoreProgram({
          transfer: AttentionScoreTransferMode::from_schedule(
            self.memory,
            stages=self.pipeline_stages,
            immediate=supports_immediate_copy &&
              matrix_probability &&
              self.memory == AttentionStagedShared &&
              self.pipeline_stages == 1 &&
              self.vector_bytes == 16,
          ),
          resident_statistics: matrix_probability &&
          self.reuse_online_softmax_tile_storage,
          resident_output: self.output_fold_storage() ==
          SubgroupResidentOutputFold,
          matrix_probability,
          factor_shared_key: factor_shared_key && queries > fragment_rows,
          query_offsets: scalarize_attention_query_fold(
            queries,
            self.subgroups_per_block,
          ),
        })
      }
    AttentionSubgroupDot => raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
}

///|
pub fn AttentionSharedScoreProgram::transfer_mode(
  self : AttentionSharedScoreProgram,
) -> AttentionScoreTransferMode {
  self.transfer
}

///|
pub fn AttentionSharedScoreProgram::resident_statistics(
  self : AttentionSharedScoreProgram,
) -> Bool {
  self.resident_statistics
}

///|
pub fn AttentionSharedScoreProgram::resident_output(
  self : AttentionSharedScoreProgram,
) -> Bool {
  self.resident_output
}

///|
pub fn AttentionSharedScoreProgram::matrix_probability(
  self : AttentionSharedScoreProgram,
) -> Bool {
  self.matrix_probability
}

///|
pub fn AttentionSharedScoreProgram::factor_shared_key(
  self : AttentionSharedScoreProgram,
) -> Bool {
  self.factor_shared_key
}

///|
pub fn AttentionSharedScoreProgram::query_offsets(
  self : AttentionSharedScoreProgram,
) -> ReadOnlyArray[Int] {
  self.query_offsets
}

///|
pub fn AttentionSharedScoreProgram::fold_steps(
  self : AttentionSharedScoreProgram,
) -> ReadOnlyArray[AttentionScoreFoldStep] {
  plan_attention_score_fold(
    resident_statistics=self.resident_statistics,
    resident_output=self.resident_output,
  )
}

///|
pub fn AttentionSharedScoreProgram::terminal(
  self : AttentionSharedScoreProgram,
) -> AttentionScoreTerminalPlan {
  AttentionScoreTerminalPlan::new(
    resident_statistics=self.resident_statistics,
    resident_output=self.resident_output,
  )
}

///|
pub fn AttentionSharedScoreProgram::bootstrap(
  self : AttentionSharedScoreProgram,
  point : AttentionScoreBootstrapPoint,
) -> ReadOnlyArray[AttentionScoreBootstrapEffect] {
  plan_attention_score_bootstrap(point, resident_output=self.resident_output)
}
