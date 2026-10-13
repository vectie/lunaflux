///|
/// A rectangular byte-plane view. Rows are contiguous within each source
/// stride; the destination is compact. This is data, not a filesystem effect.
pub struct PackedPlaneSlice {
  priv source_offset : Int64
  priv source_stride : Int64
  priv rows : Int
  priv row_bytes : Int64
}

///|
pub fn PackedPlaneSlice::geometry(self : Self) -> (Int64, Int64, Int, Int64) {
  (self.source_offset, self.source_stride, self.rows, self.row_bytes)
}

///|
/// An exact packed submatrix. Sub-byte payloads and scale blocks may not be
/// split. No re-quantization, decoding, or change of global scale is implied.
pub struct PackedMatrixSlice {
  priv source : @format.Representation
  priv representation : @format.Representation
  priv payload : PackedPlaneSlice
  priv scales : PackedPlaneSlice?
}

///|
pub fn PackedMatrixSlice::new(
  source : @format.Representation,
  row_start~ : Int,
  rows~ : Int,
  column_start~ : Int,
  columns~ : Int,
) -> Self raise PrecisionPlanError {
  let (source_rows, source_columns) = source.shape()
  guard row_start >= 0 &&
    column_start >= 0 &&
    rows > 0 &&
    columns > 0 &&
    rows <= source_rows &&
    row_start <= source_rows - rows &&
    columns <= source_columns &&
    column_start <= source_columns - columns else {
    raise NoImplementation
  }
  let bits = source.element().bits().to_int64()
  guard column_start.to_int64() * bits % 8L == 0L &&
    (
      columns.to_int64() * bits % 8L == 0L ||
      column_start + columns == source_columns
    ) else {
    raise NoImplementation
  }
  let scales = match source.grouping() {
    Absent => None
    Tensor =>
      Some({
        source_offset: 0L,
        source_stride: source.scale_bytes(),
        rows: 1,
        row_bytes: source.scale_bytes(),
      })
    Row => {
      let bytes = source.scale_element().bytes().to_int64()
      Some({
        source_offset: row_start.to_int64() * bytes,
        source_stride: bytes,
        rows,
        row_bytes: bytes,
      })
    }
    Block(rows=block_rows, columns=block_columns) => {
      guard row_start % block_rows == 0 &&
        column_start % block_columns == 0 &&
        (rows % block_rows == 0 || row_start + rows == source_rows) &&
        (
          columns % block_columns == 0 ||
          column_start + columns == source_columns
        ) else {
        raise NoImplementation
      }
      let (_, scale_columns) = source.scale_shape()
      let bytes = source.scale_element().bytes().to_int64()
      let stride = scale_columns.to_int64() * bytes
      Some({
        source_offset: (row_start / block_rows).to_int64() * stride +
        (column_start / block_columns).to_int64() * bytes,
        source_stride: stride,
        rows: (rows - 1) / block_rows + 1,
        row_bytes: ((columns - 1) / block_columns + 1).to_int64() * bytes,
      })
    }
  }
  let representation = @format.Representation::new(
    format=source.format(),
    rows~,
    columns~,
    grouping=source.grouping(),
  ) catch {
    _ => raise NoImplementation
  }
  {
    source,
    representation,
    scales,
    payload: {
      source_offset: row_start.to_int64() * source.row_bytes() +
      column_start.to_int64() * bits / 8L,
      source_stride: source.row_bytes(),
      rows,
      row_bytes: representation.row_bytes(),
    },
  }
}

///|
pub fn PackedMatrixSlice::source(self : Self) -> @format.Representation {
  self.source
}

///|
pub fn PackedMatrixSlice::representation(self : Self) -> @format.Representation {
  self.representation
}

///|
pub fn PackedMatrixSlice::payload(self : Self) -> PackedPlaneSlice {
  self.payload
}

///|
pub fn PackedMatrixSlice::scales(self : Self) -> PackedPlaneSlice? {
  self.scales
}
