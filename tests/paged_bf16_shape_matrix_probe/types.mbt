///|
suberror ShapeMatrixFailure {
  ShapeMatrixFailure(String)
}

///|
fn require_matrix(
  condition : Bool,
  message : String,
) -> Unit raise ShapeMatrixFailure {
  if !condition {
    raise ShapeMatrixFailure(message)
  }
}

///|
priv struct ShapeMatrixCase {
  name : String
  counts : FixedArray[Int]
  token_ids : FixedArray[Int]
  positions : FixedArray[Int]
  row_offsets : FixedArray[Int]
  sequence_lengths : FixedArray[Int]
  page_offsets : FixedArray[Int]
  page_indices : FixedArray[Int]
}

///|
fn shape_matrix_case(
  name : String,
  counts : FixedArray[Int],
  token_ids : FixedArray[Int],
  positions : FixedArray[Int],
  row_offsets : FixedArray[Int],
  sequence_lengths : FixedArray[Int],
  page_offsets : FixedArray[Int],
  page_indices : FixedArray[Int],
) -> ShapeMatrixCase {
  {
    name,
    counts,
    token_ids,
    positions,
    row_offsets,
    sequence_lengths,
    page_offsets,
    page_indices,
  }
}

///|
fn shape_matrix_cases() -> FixedArray[ShapeMatrixCase] {
  [
    shape_matrix_case(
      "prefill-single",
      [1, 0, 1, 1, 1],
      [2, 0, 0],
      [0, 0, 0],
      [0, 1, 1],
      [1, 0],
      [0, 1, 1],
      [0, 0, 0],
    ),
    shape_matrix_case(
      "prefill-page-edge",
      [1, 0, 1, 2, 1],
      [5, 7, 0],
      [0, 1, 0],
      [0, 2, 2],
      [2, 0],
      [0, 1, 1],
      [2, 0, 0],
    ),
    shape_matrix_case(
      "decode-cached-prefix",
      [0, 1, 1, 1, 2],
      [9, 0, 0],
      [3, 0, 0],
      [0, 1, 1],
      [4, 0],
      [0, 2, 2],
      [1, 3, 0],
    ),
    shape_matrix_case(
      "mixed-prefill-decode",
      [1, 1, 2, 3, 3],
      [2, 5, 7],
      [0, 1, 2],
      [0, 2, 3],
      [2, 3],
      [0, 1, 3],
      [0, 1, 2],
    ),
    shape_matrix_case(
      "two-decode-rows",
      [0, 2, 2, 2, 3],
      [4, 11, 0],
      [1, 3, 0],
      [0, 1, 2],
      [2, 4],
      [0, 1, 3],
      [0, 1, 3],
    ),
  ]
}

///|
fn ShapeMatrixCase::live_tokens(self : ShapeMatrixCase) -> Int {
  self.counts[3]
}

///|
fn ShapeMatrixCase::row_for_token(self : ShapeMatrixCase, token : Int) -> Int {
  if self.counts[2] == 2 && token >= self.row_offsets[1] {
    1
  } else {
    0
  }
}
