///|
/// Borrowed, preallocated batch page ports. This owner does not allocate KV or
/// interpret request identities; the scheduler owns page lifetime and the
/// batch frame copies its exact IDs/generations before device submission.
#valtype
pub struct SparseHistoryPagePorts {
  priv rows : @device.Allocation
  priv offsets : @device.Allocation
  priv pages : @device.Allocation
  priv generations : @device.Allocation
}

///|
pub fn SparseHistoryPagePorts::new(
  row_sequences~ : @device.Allocation,
  page_offsets~ : @device.Allocation,
  page_ids~ : @device.Allocation,
  page_generations~ : @device.Allocation,
) -> Self {
  {
    rows: row_sequences,
    offsets: page_offsets,
    pages: page_ids,
    generations: page_generations,
  }
}

///|
fn page_arguments(
  plan : @precision.IndexedSparseAttentionPrecision,
  ports : SparseHistoryPagePorts?,
) -> Array[@device.KernelArgument] raise @device.DeviceError {
  match (plan.paged_history(), ports) {
    (None, None) => []
    (Some(domain), Some(ports)) => {
      let (_, _, _, sequences) = domain.geometry()
      let pages = domain.table_capacity().to_int64() * 4L
      [
        region(ports.rows, plan.capacity().0.to_int64() * 4L, 4L),
        region(ports.offsets, (sequences.to_int64() + 1L) * 4L, 4L),
        region(ports.pages, pages, 4L),
        region(ports.generations, pages, 4L),
      ]
    }
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
}
