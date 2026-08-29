///|
pub(all) enum Fp8LaunchScope {
  StatelessFullPrefillV1
  PagedFullGraphV1
} derive(Debug, Eq)

///|
pub struct Fp8PagedRowShape {
  priv query_tokens : Int
  priv context_tokens : Int
  priv page_table_entries : Int
} derive(Debug, Eq)

///|
pub fn Fp8PagedRowShape::new(
  query_tokens~ : Int,
  context_tokens~ : Int,
  page_table_entries~ : Int,
) -> Fp8PagedRowShape raise Fp8LaunchAbiError {
  if query_tokens <= 0 ||
    context_tokens < query_tokens ||
    page_table_entries <= 0 {
    raise InvalidFp8LaunchAbi(InvalidShape)
  }
  { query_tokens, context_tokens, page_table_entries }
}

///|
pub fn Fp8PagedRowShape::query_tokens(self : Fp8PagedRowShape) -> Int {
  self.query_tokens
}

///|
pub fn Fp8PagedRowShape::context_tokens(self : Fp8PagedRowShape) -> Int {
  self.context_tokens
}

///|
pub fn Fp8PagedRowShape::page_table_entries(self : Fp8PagedRowShape) -> Int {
  self.page_table_entries
}

///|
priv enum Fp8ExactExecutionShapeData {
  Stateless(
    @launch_contract.KernelProfileId,
    @device_profile.ExactExecutionShape
  )
  Paged(
    @launch_contract.KernelProfileId,
    ReadOnlyArray[Fp8PagedRowShape],
    Int,
    Int
  )
} derive(Debug, Eq)

///|
pub struct Fp8ExactExecutionShape {
  priv data : Fp8ExactExecutionShapeData
} derive(Debug, Eq)

///|
pub fn Fp8ExactExecutionShape::stateless_full_prefill(
  profile~ : @launch_contract.KernelProfileId,
  shape : @device_profile.ExactExecutionShape,
) -> Fp8ExactExecutionShape raise Fp8LaunchAbiError {
  if shape.phase() != FullPrefill {
    raise InvalidFp8LaunchAbi(InvalidShape)
  }
  { data: Stateless(profile, shape) }
}

///|
pub fn Fp8ExactExecutionShape::paged_full_graph(
  profile~ : @launch_contract.KernelProfileId,
  rows~ : ArrayView[Fp8PagedRowShape],
) -> Fp8ExactExecutionShape raise Fp8LaunchAbiError {
  if rows.is_empty() || rows.length() > 65536 {
    raise InvalidFp8LaunchAbi(InvalidShape)
  }
  let mut query_tokens = 0L
  let mut page_table_entries = 0L
  for row in rows {
    let query = row.query_tokens.to_int64()
    let pages = row.page_table_entries.to_int64()
    if query_tokens > 2147483647L - query ||
      page_table_entries > 2147483647L - pages {
      raise InvalidFp8LaunchAbi(ShapeOverflow)
    }
    query_tokens += query
    page_table_entries += pages
  }
  {
    data: Paged(
      profile,
      ReadOnlyArray::from_array(rows.to_owned()),
      query_tokens.to_int(),
      page_table_entries.to_int(),
    ),
  }
}

///|
pub fn Fp8ExactExecutionShape::scope(
  self : Fp8ExactExecutionShape,
) -> Fp8LaunchScope {
  match self.data {
    Stateless(_, _) => StatelessFullPrefillV1
    Paged(_, _, _, _) => PagedFullGraphV1
  }
}

///|
pub fn Fp8ExactExecutionShape::profile(
  self : Fp8ExactExecutionShape,
) -> @launch_contract.KernelProfileId? {
  match self.data {
    Stateless(profile, _) => Some(profile)
    Paged(profile, _, _, _) => Some(profile)
  }
}

///|
pub fn Fp8ExactExecutionShape::batch_rows(self : Fp8ExactExecutionShape) -> Int {
  match self.data {
    Stateless(_, shape) => shape.batch_rows()
    Paged(_, rows, _, _) => rows.length()
  }
}

///|
pub fn Fp8ExactExecutionShape::query_tokens(
  self : Fp8ExactExecutionShape,
) -> Int64 {
  match self.data {
    Stateless(_, shape) => shape.token_rows()
    Paged(_, _, query_tokens, _) => query_tokens.to_int64()
  }
}

///|
pub fn Fp8ExactExecutionShape::paged_rows(
  self : Fp8ExactExecutionShape,
) -> ReadOnlyArray[Fp8PagedRowShape]? {
  match self.data {
    Stateless(_, _) => None
    Paged(_, rows, _, _) => Some(rows)
  }
}
