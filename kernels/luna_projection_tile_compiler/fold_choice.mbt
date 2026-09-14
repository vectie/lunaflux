///|
/// Independent ordered folds in the semantic projection graph.
pub(all) enum ProjectionFoldRole {
  MatrixFold
  SiblingFold
  IntermediateFold
} derive(Debug, Eq)

///|
pub struct ProjectionFoldChoice {
  priv role : ProjectionFoldRole
  priv stages : Int
  priv transfer_tiles : Int
  priv fragment_stages : Int
} derive(Debug, Eq)

///|
pub fn ProjectionFoldChoice::new(
  role : ProjectionFoldRole,
  stages~ : Int,
  transfer_tiles~ : Int,
  fragment_stages~ : Int,
) -> ProjectionFoldChoice raise ProjectionTileCompilerError {
  guard stages >= 2 &&
    stages <= 4 &&
    [1, 2, 4, 8, 16].contains(transfer_tiles) &&
    fragment_stages >= 1 &&
    fragment_stages <= 2 &&
    fragment_stages <= transfer_tiles else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  { role, stages, transfer_tiles, fragment_stages, }
}

///|
pub fn ProjectionTileCompilerRequest::with_fold_choices(
  self : ProjectionTileCompilerRequest,
  choices : ArrayView[ProjectionFoldChoice],
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  let mut request = self
  let seen : Array[ProjectionFoldRole] = []
  for choice in choices {
    guard !seen.contains(choice.role) else {
      raise InvalidProjectionTileCompilation(InvalidProblem)
    }
    seen.push(choice.role)
    request = request.with_fold_pipeline(
      choice.role,
      stages=choice.stages,
      transfer_tiles=choice.transfer_tiles,
      fragment_stages=choice.fragment_stages,
    )
  }
  request
}

///|
/// Select an offline scheduling alternative without changing the pure program
/// or its ordered accumulation. Unsupported folds and incomplete transfers are
/// rejected before lowering. Device resource feasibility is checked by the AOT
/// backend and offline compiler, not by model semantics.
pub fn ProjectionTileCompilerRequest::with_fold_pipeline(
  self : ProjectionTileCompilerRequest,
  role : ProjectionFoldRole,
  stages~ : Int,
  transfer_tiles~ : Int,
  fragment_stages~ : Int,
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  guard stages >= 2 &&
    stages <= 4 &&
    [1, 2, 4, 8, 16].contains(transfer_tiles) &&
    fragment_stages >= 1 &&
    fragment_stages <= 2 else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  let base = compile_projection_tile({ ..self, fold_choices: [], }).schedule()
  let pipeline = match role {
    MatrixFold => base.matrix_pipeline()
    SiblingFold => base.sibling_pipeline()
    IntermediateFold => base.intermediate_pipeline()
  }
  guard pipeline is Some(pipeline) &&
    pipeline.tile_count() % transfer_tiles == 0 &&
    pipeline.tile_count() / transfer_tiles >= stages &&
    fragment_stages <= transfer_tiles else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  let choice = { role, stages, transfer_tiles, fragment_stages, }
  // Canonical role order makes composition independent of configuration order.
  let choices : Array[ProjectionFoldChoice] = []
  for kind in [MatrixFold, SiblingFold, IntermediateFold] {
    if kind == role {
      choices.push(choice)
    } else {
      for prior in self.fold_choices {
        if prior.role == kind {
          choices.push(prior)
        }
      }
    }
  }
  { ..self, fold_choices: ReadOnlyArray::from_array(choices), }
}

///|
fn projection_apply_fold_choice(
  base : ProjectionFoldPipeline,
  role : ProjectionFoldRole,
  request : ProjectionTileCompilerRequest,
) -> ProjectionFoldPipeline {
  for choice in request.fold_choices {
    if choice.role == role {
      return {
        ..base,
        stages: choice.stages,
        transfer_width: base.tile_width * choice.transfer_tiles,
        fragments: Some(choice.fragment_stages),
      }
    }
  }
  base
}

///|
/// One transfer is consumed only after publication and its slot is not reused
/// until that fold iteration has completed. These scalar facts are shared by
/// terminal lowerers and the finite lifetime-model tests.
pub fn ProjectionFoldPipeline::transfer_count(
  self : ProjectionFoldPipeline,
) -> Int {
  self.tile_count * self.tile_width / self.transfer_width
}

///|
pub fn ProjectionFoldPipeline::slot(
  self : ProjectionFoldPipeline,
  transfer : Int,
) -> Int? {
  if transfer < 0 || transfer >= self.transfer_count() {
    None
  } else {
    Some(transfer % self.stages)
  }
}
