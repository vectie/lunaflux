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
  priv row_tiles : Int?
  priv output_tiles : Int?
  priv hoist_addresses : Bool
  priv consumer_groups : Int?
  priv maximum_token_rows : Int?
} derive(Debug, Eq)

///|
pub fn ProjectionFoldChoice::new(
  role : ProjectionFoldRole,
  stages~ : Int,
  transfer_tiles~ : Int,
  fragment_stages~ : Int,
  row_tiles? : Int,
  output_tiles? : Int,
  hoist_addresses? : Bool = false,
  consumer_groups? : Int,
  maximum_token_rows? : Int,
) -> ProjectionFoldChoice raise ProjectionTileCompilerError {
  guard stages >= 2 &&
    stages <= 4 &&
    [1, 2, 4, 8, 16].contains(transfer_tiles) &&
    fragment_stages >= 1 &&
    fragment_stages <= 2 &&
    fragment_stages <= transfer_tiles &&
    row_tiles.map(n => n >= 1 && n <= 16).unwrap_or(true) &&
    output_tiles.map(n => n >= 1 && n <= 32).unwrap_or(true) &&
    consumer_groups.map(n => [1, 2, 4, 8, 16, 32].contains(n)).unwrap_or(true) &&
    maximum_token_rows.map(n => n >= 2 && n <= 1048576).unwrap_or(true) else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  {
    role,
    stages,
    transfer_tiles,
    fragment_stages,
    row_tiles,
    output_tiles,
    hoist_addresses,
    consumer_groups,
    maximum_token_rows,
  }
}

///|
pub fn ProjectionTileCompilerRequest::with_fold_choices(
  self : ProjectionTileCompilerRequest,
  choices : ArrayView[ProjectionFoldChoice],
) -> ProjectionTileCompilerRequest raise ProjectionTileCompilerError {
  let mut request = self
  let seen : Array[(ProjectionFoldRole, Int?)] = []
  for choice in choices {
    let key = (choice.role, choice.maximum_token_rows)
    guard !seen.contains(key) else {
      raise InvalidProjectionTileCompilation(InvalidProblem)
    }
    seen.push(key)
  }
  // Select the narrowest applicable row domain before device lowering. This
  // preserves a measured large-row schedule while refining small-row products;
  // input order cannot change the selected executable.
  for role in [MatrixFold, SiblingFold, IntermediateFold] {
    let mut selected : ProjectionFoldChoice? = None
    for choice in choices {
      if choice.role != role ||
        choice.maximum_token_rows
        .map(n => self.maximum_tokens > n)
        .unwrap_or(false) {
        continue
      }
      if selected is None ||
        choice.maximum_token_rows.unwrap_or(1048577) <
        selected.unwrap().maximum_token_rows.unwrap_or(1048577) {
        selected = Some(choice)
      }
    }
    guard selected is Some(choice) else { continue }
    request = request.with_fold_pipeline(
      choice.role,
      stages=choice.stages,
      transfer_tiles=choice.transfer_tiles,
      fragment_stages=choice.fragment_stages,
      row_tiles?=choice.row_tiles,
      output_tiles?=choice.output_tiles,
      hoist_addresses=choice.hoist_addresses,
      consumer_groups?=choice.consumer_groups,
    )
  }
  request
}

///|
pub fn ProjectionFoldChoice::maximum_token_rows(
  self : ProjectionFoldChoice,
) -> Int? {
  self.maximum_token_rows
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
  row_tiles? : Int,
  output_tiles? : Int,
  hoist_addresses? : Bool = false,
  consumer_groups? : Int,
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
  // Geometry belongs to independent maps, never the reduction axis. Complete
  // ingress heads have separate refinement APIs. Sibling ownership is refined
  // independently from its row/output product; the materialized down wrapper
  // retains its own fixed column envelope.
  guard (row_tiles is None || self.epilogue == ProjectionStoreRows) &&
    row_tiles.map(n => n >= 1 && n <= 16).unwrap_or(true) &&
    output_tiles
    .map(n => {
      n >= 1 && n <= self.dispatch.prefill().output_tiles_per_workgroup()
    })
    .unwrap_or(true) &&
    (!hoist_addresses || role != SiblingFold) &&
    consumer_groups
    .map(groups => {
      [1, 2, 4, 8, 16, 32].contains(groups) &&
      groups <= self.dispatch.prefill().output_tiles_per_workgroup() &&
      groups >= output_tiles.unwrap_or(pipeline.output_tiles()) &&
      (
        role != IntermediateFold ||
        groups == output_tiles.unwrap_or(pipeline.output_tiles())
      )
    })
    .unwrap_or(true) else {
    raise InvalidProjectionTileCompilation(InvalidProblem)
  }
  let choice = {
    role,
    stages,
    transfer_tiles,
    fragment_stages,
    row_tiles,
    output_tiles,
    hoist_addresses,
    consumer_groups,
    maximum_token_rows: None,
  }
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
        row_width: match choice.row_tiles {
          Some(tiles) =>
            projection_bounded_row_extent(
              maximum_rows=if request.demand == QueryRowEnds {
                request.maximum_rows
              } else {
                request.maximum_tokens
              },
              matrix_rows=base.tile_width,
              preferred_rows=tiles * base.tile_width,
            ).unwrap()
          None => base.row_width
        },
        output_tiles: choice.output_tiles.unwrap_or(base.output_tiles),
        hoist_addresses: choice.hoist_addresses,
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
