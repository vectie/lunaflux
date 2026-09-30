///|
/// Immutable final refinement shared by every projection emitter. The device
/// supplies layout and transfer capabilities; semantic geometry and numerical
/// laws are taken from the scheduled program, not reconstructed from strings.
pub struct PhysicalProjectionTileCompilation {
  scheduled : ScheduledProjectionTileCompilation
  scalar : ProjectionScalarFoldPlan
  sibling_scalar : ProjectionScalarFoldPlan?
  down_scalar : ProjectionScalarFoldPlan?
  ordered_down : ProjectionScalarFoldPlan?
  matrix : ProjectionPhysicalPipeline?
  sibling : ProjectionPhysicalPipeline?
  sibling_epilogue : ProjectionSiblingEpiloguePlan?
  intermediate : ProjectionPhysicalPipeline?
  mapped : (@physical_tile_ir.FragmentProgram, @physical_tile_ir.OperandLayout)?
  map_effects : ProjectionMapEffectPlan
  selected_gather : (ProjectionSelectedRowGather, ProjectionOperandWindow)?
  weight_selection : ProjectionRowSelection
  matrix_weights : ProjectionWeightTransferPlan?
  materialization : ProjectionMaterializationEffects
} derive(Debug, Eq)

///|
pub extend PhysicalProjectionTileCompilation with @debug.Debug::{to_repr}

///|
pub extend PhysicalProjectionTileCompilation with Eq::{equal, not_equal}

///|
pub fn ScheduledProjectionTileCompilation::refine_physical(
  self : ScheduledProjectionTileCompilation,
  layout : (Int) -> @physical_tile_ir.OperandLayout?,
  execution : (Int) -> ProjectionTransferExecution,
  owners_per_group~ : Int,
  sibling_ownership? : ProjectionSiblingOwnership = SplitSiblingPlanes,
  sibling_transport? : ProjectionSiblingTransport = BaselineSiblingTransport,
) -> PhysicalProjectionTileCompilation? {
  guard owners_per_group > 0 else { return None }
  let schedule = self.schedule()
  let (input_width, output_width, intermediate_width) = self.program().shape.dimensions()
  let reduction = schedule.single_row_reduction()
  guard ProjectionScalarFoldPlan::new(extent=input_width, products=1, reduction)
    is Some(scalar) else {
    return None
  }
  let sibling_scalar = if intermediate_width > 0 {
    ProjectionScalarFoldPlan::new(extent=input_width, products=2, reduction)
  } else {
    None
  }
  let down_scalar = if intermediate_width > 0 {
    ProjectionScalarFoldPlan::new(
      extent=intermediate_width,
      products=4,
      reduction,
    )
  } else {
    None
  }
  let ordered_down = if intermediate_width > 0 {
    ProjectionScalarFoldPlan::new(
      extent=intermediate_width,
      products=1,
      OrderedDot,
    )
  } else {
    None
  }
  fn bind(
    pipeline : ProjectionFoldPipeline?,
    groups : Int,
    products : Int,
  ) -> ProjectionPhysicalPipeline? {
    guard pipeline is Some(pipeline) else { return None }
    guard layout(pipeline.transfer_width()) is Some(storage) else {
      return None
    }
    ProjectionPhysicalPipeline::new(
      pipeline,
      storage,
      consumer_groups=groups,
      owners_per_group~,
      products~,
      execution(pipeline.transfer_width()),
      sibling_ownership~,
      sibling_transport~,
    )
  }
  let groups = schedule.output_tiles_per_workgroup()
  let matrix = bind(schedule.matrix_pipeline(), groups, 1)
  let sibling = bind(schedule.sibling_pipeline(), groups, 2)
  let sibling_epilogue = match sibling {
    Some(pipeline) => pipeline.sibling_epilogue(sibling_ownership)
    None => None
  }
  let intermediate = bind(
    schedule.intermediate_pipeline(),
    schedule.companion_output_tiles_per_workgroup().unwrap_or(groups),
    1,
  )
  guard (schedule.matrix_pipeline() is None || matrix is Some(_)) &&
    (
      schedule.sibling_pipeline() is None ||
      (sibling is Some(_) && sibling_epilogue is Some(_))
    ) &&
    (schedule.intermediate_pipeline() is None || intermediate is Some(_)) else {
    return None
  }
  let mapped = match
    (if schedule.row_tail_policy() == ZeroPaddedMatrixRowTail {
      schedule.mapped_fragment_program(input_width)
    } else {
      None
    }) {
    Some(fragment) => {
      guard layout(fragment.geometry().2) is Some(storage) else { return None }
      guard storage.geometry().0 == fragment.geometry().2 else { return None }
      Some((fragment, storage))
    }
    None => None
  }
  let selected_gather = match self.selected_row_gather() {
    Some(gather) => {
      guard gather.operand_window(input_width) is Some(window) else {
        return None
      }
      Some((gather, window))
    }
    None => None
  }
  let segments = if self.program().concatenated_weights().is_empty() {
    plan_projection_concatenated_rows([output_width]).unwrap()
  } else {
    self.program().concatenated_weights()
  }
  guard plan_projection_row_selection(segments) is Some(weight_selection) else {
    return None
  }
  let matrix_weights = match matrix {
    Some(physical) => {
      let width = physical.schedule.transfer_width()
      let vectors = physical.operands.region(1).unwrap().ownership().geometry()
      guard ProjectionVectorTransferMap::new(
          vectors=vectors.0,
          owners=vectors.1,
        )
        is Some(map) else {
        return None
      }
      ProjectionWeightTransferPlan::new(
        map,
        segments,
        row_width=width,
        vector_width=physical.operands.layout().geometry().1,
      )
    }
    None => None
  }
  guard matrix is None || matrix_weights is Some(_) else { return None }
  Some({
    scheduled: self,
    scalar,
    sibling_scalar,
    down_scalar,
    ordered_down,
    matrix,
    sibling,
    sibling_epilogue,
    intermediate,
    mapped,
    selected_gather,
    weight_selection,
    matrix_weights,
    map_effects: schedule.demand().map_effect_plan(),
    materialization: WorkgroupStorage.materialization_effects(
      reused=sibling is Some(p) && p.sibling_reuse,
    ),
  })
}

///|
pub fn PhysicalProjectionTileCompilation::schedule(
  self : PhysicalProjectionTileCompilation,
) -> ProjectionTileSchedule {
  self.scheduled.schedule()
}

///|
pub fn PhysicalProjectionTileCompilation::program(
  self : PhysicalProjectionTileCompilation,
) -> ProjectionTileProgram {
  self.scheduled.program()
}
