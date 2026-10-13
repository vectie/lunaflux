///|
/// Read a strided plane in bounded sequential windows, compact each window in
/// place, then publish contiguous destination bytes. Gaps are read but never
/// published. This avoids millions of per-row requests/seeks for column TP.
/// Compacting cannot overwrite unread bytes because write <= read throughout.
fn transfer_strided_file(
  shard : Int,
  file : @approved_fs.ApprovedFile,
  request_index : Int,
  request : StreamingSafetensorsTransfer,
  start : Int64,
  scratch : FixedArray[Byte],
  consume : (Int, Int64, FixedArray[Byte], Int) -> Bool,
) -> Unit raise StreamingSafetensorsError {
  let stride = request.source_row_stride
  let width = request.byte_count / request.row_count.to_int64()
  let extent = (request.row_count - 1).to_int64() * stride + width
  for scanned = 0L; scanned < extent; {
    let count = scratch.length().to_int64().min(extent - scanned).to_int()
    read_exactly(shard, file, scratch, 0, count, start + scanned)
    let mut read = 0
    let mut written = 0
    let mut destination = 0L
    while read < count {
      let position = scanned + read.to_int64()
      let row = position / stride
      let column = position % stride
      if column >= width {
        read += (stride - column).min((count - read).to_int64()).to_int()
      } else {
        let size = (width - column).min((count - read).to_int64()).to_int()
        if written == 0 {
          destination = row * width + column
        }
        for i in 0..<size {
          scratch[written + i] = scratch[read + i]
        }
        written += size
        read += size
      }
    }
    if written > 0 &&
      !consume(
        request.destination_ordinal,
        request.destination_offset + destination,
        scratch,
        written,
      ) {
      raise VisitorRejected(request_index)
    }
    continue scanned + count.to_int64()
  }
}
