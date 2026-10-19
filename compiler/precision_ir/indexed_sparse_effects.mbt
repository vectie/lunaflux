///|
/// External projected Q/K/V, index inputs and history precede pooling, stable
/// selection and attention. No comparison data participates in these effects.
pub fn IndexedSparseAttentionPrecision::program(
  self : Self,
) -> @parallel.Program raise PrecisionPlanError {
  let history = self.physical_history().to_int64()
  let rows = self.queries.to_int64()
  let index_row = self.index_dimension.to_int64() * 2L
  let qk = self.qk_dimension.to_int64() * 2L
  let values = self.value_dimension.to_int64() * 2L
  let sizes = [
    20L,
    20L,
    rows * 4L,
    history * 4L,
    history * index_row,
    history * index_row,
    self.pool_size.to_int64() * index_row,
    rows * self.index_heads.to_int64() * index_row,
    rows * self.index_heads.to_int64() * 2L,
    rows * self.query_heads.to_int64() * qk,
    history * self.kv_heads.to_int64() * qk,
    history * self.kv_heads.to_int64() * values,
    self.frame_sizes[0],
    self.frame_sizes[1],
    self.frame_sizes[2],
    self.frame_sizes[3],
    self.frame_sizes[4],
    rows * self.query_heads.to_int64() * values,
  ]
  let initialized = [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    self.paging is Some(_),
    self.paging is Some(_),
    self.paging is Some(_),
    false,
    false,
    false,
  ]
  let stages : Array[Array[(Int, @parallel.AccessMode)]] = [
    [
      (1, Read),
      (0, Read),
      (2, Read),
      (3, Read),
      (4, Read),
      (5, Read),
      (6, Read),
      (12, if self.paging is Some(_) { ReadWrite } else { Write }),
      (13, if self.paging is Some(_) { ReadWrite } else { Write }),
      (14, if self.paging is Some(_) { ReadWrite } else { Write }),
      (15, Write),
    ],
    [
      (0, Read),
      (2, Read),
      (3, Read),
      (7, Read),
      (8, Read),
      (12, Read),
      (13, Read),
      (14, Read),
      (15, Read),
      (16, Write),
    ],
    [
      (0, Read),
      (1, Read),
      (2, Read),
      (3, Read),
      (9, Read),
      (10, Read),
      (11, Read),
      (16, Read),
      (17, Write),
    ],
  ]
  if self.paging is Some(domain) {
    let sequences = domain.geometry().3.to_int64()
    sizes.append([
      rows * 4L,
      (sequences + 1L) * 4L,
      domain.table_capacity().to_int64() * 4L,
      domain.table_capacity().to_int64() * 4L,
    ])
    initialized.append([true, true, true, true])
    for stage in stages {
      stage.append([(18, Read), (19, Read), (20, Read), (21, Read)])
    }
  }
  numeric_effects(sizes[:], initialized[:], stages[:])
}
