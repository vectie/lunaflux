///|
fn evaluate_selection(
  tree : @luna_projection_tile_compiler.ProjectionRowSelection,
  row : Int,
) -> (Int, Int) {
  match tree {
    Operand(operand) => (operand, row)
    Skip(width~, operand~, next) =>
      if row < width {
        (operand, row)
      } else {
        evaluate_selection(next, row - width)
      }
  }
}

///|
test "ordered operand selection preserves every row and vector worker mapping" {
  for counts in [[17], [8, 2, 2], [3, 5, 7, 11], Array::make(129, 3)] {
    let segments = @luna_projection_tile_compiler.plan_projection_concatenated_rows(
      counts,
    ).unwrap()
    let tree = @luna_projection_tile_compiler.plan_projection_row_selection(
      segments,
    ).unwrap()
    for segment in segments {
      for row in segment.row_begin..<segment.row_end {
        assert_eq(
          evaluate_selection(tree, row),
          (segment.operand, row - segment.row_begin),
        )
      }
    }
    let rows = segments[segments.length() - 1].row_end
    for owners in [32, 64, 128, 256] {
      let map = @luna_projection_tile_compiler.ProjectionVectorTransferMap::new(
        vectors=rows * 8,
        owners~,
      ).unwrap()
      let (_, _, rounds) = map.geometry()
      for owner in 0..<owners {
        for round in 0..<rounds {
          if map.vector_at(owner, round) is Some(vector) {
            let (operand, local_row) = evaluate_selection(tree, vector / 8)
            assert_eq(segments[operand].row_begin + local_row, vector / 8)
          }
        }
      }
    }
  }
  assert_true(
    @luna_projection_tile_compiler.plan_projection_row_selection([]) is None,
  )
}
