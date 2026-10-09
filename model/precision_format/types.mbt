///|
/// Logical formats, not device instructions or checkpoint-specific packing.
pub(all) enum Format {
  Bf16
  Fp16
  F32
  Fp8E4M3
  Fp8E5M2
  Int8
  Int4
  Fp4E2M1
  NvFp4
  MxFp4
  MxFp8E4M3
  MxFp8E5M2
} derive(Debug, Eq)

///|
pub(all) enum Element {
  Bf16
  Fp16
  F32
  E4M3
  E5M2
  Signed8
  Signed4
  E2M1
} derive(Debug, Eq)

///|
pub(all) enum ScaleElement {
  Absent
  F32
  Ue8M0
  E4M3
} derive(Debug, Eq)

///|
pub(all) enum ScaleGrouping {
  Absent
  Tensor
  Row
  Block(rows~ : Int, columns~ : Int)
} derive(Debug, Eq)

///|
pub(all) suberror PrecisionError {
  InvalidShape
  InvalidGrouping
  SizeOverflow
  InvalidCode(Int)
  NonFiniteValue
  InvalidScale
  PayloadMismatch
  InvalidPadding
  IndexOutOfBounds
  OutputTooSmall
} derive(Debug, Eq)

///|
/// Immutable representation semantics. Four-bit payloads are row-aligned,
/// low logical column in the low nibble; unused high tail nibbles are zero.
/// Scales are logical row-major; hardware swizzles belong to device lowering.
pub struct Representation {
  priv format : Format
  priv element : Element
  priv rows : Int
  priv columns : Int
  priv grouping : ScaleGrouping
  priv scale_element : ScaleElement
  priv global_scale : Bool
  priv row_bytes : Int64
  priv payload_bytes : Int64
  priv scale_rows : Int
  priv scale_columns : Int
  priv scale_bytes : Int64
} derive(Debug, Eq)

///|
pub fn Format::element(self : Format) -> Element {
  match self {
    Bf16 => Bf16
    Fp16 => Fp16
    F32 => F32
    Fp8E4M3 | MxFp8E4M3 => E4M3
    Fp8E5M2 | MxFp8E5M2 => E5M2
    Int8 => Signed8
    Int4 => Signed4
    Fp4E2M1 | NvFp4 | MxFp4 => E2M1
  }
}

///|
pub fn Element::bits(self : Element) -> Int {
  match self {
    Bf16 | Fp16 => 16
    F32 => 32
    E4M3 | E5M2 | Signed8 => 8
    Signed4 | E2M1 => 4
  }
}

///|
pub fn Format::name(self : Format) -> String {
  match self {
    Bf16 => "bf16"
    Fp16 => "fp16"
    F32 => "f32"
    Fp8E4M3 => "fp8-e4m3fn"
    Fp8E5M2 => "fp8-e5m2"
    Int8 => "int8-symmetric"
    Int4 => "int4-symmetric"
    Fp4E2M1 => "fp4-e2m1"
    NvFp4 => "nvfp4"
    MxFp4 => "mxfp4"
    MxFp8E4M3 => "mxfp8-e4m3"
    MxFp8E5M2 => "mxfp8-e5m2"
  }
}

///|
pub fn ScaleElement::bytes(self : ScaleElement) -> Int {
  match self {
    Absent => 0
    F32 => 4
    Ue8M0 | E4M3 => 1
  }
}

///|
/// INT8/INT4 are signed two's complement, symmetric full ranges [-128,127]
/// and [-8,7]. This is distinct from legacy I8's reserved -128 encoding.
/// The default integer group is one output row, never a model-specific rule.
pub fn Representation::new(
  format~ : Format,
  rows~ : Int,
  columns~ : Int,
  grouping? : ScaleGrouping,
) -> Representation raise PrecisionError {
  if rows <= 0 || columns <= 0 {
    raise InvalidShape
  }
  let (default_grouping, scale_element, global_scale) = match format {
    Bf16 | Fp16 | F32 => (ScaleGrouping::Absent, ScaleElement::Absent, false)
    NvFp4 => (Block(rows=1, columns=16), ScaleElement::E4M3, true)
    MxFp4 | MxFp8E4M3 | MxFp8E5M2 =>
      (Block(rows=1, columns=32), ScaleElement::Ue8M0, false)
    _ => (ScaleGrouping::Row, ScaleElement::F32, false)
  }
  let grouping = grouping.unwrap_or(default_grouping)
  if (
      format == Bf16 ||
      format == Fp16 ||
      format == F32 ||
      format == NvFp4 ||
      format == MxFp4
    ) &&
    grouping != default_grouping {
    raise InvalidGrouping
  }
  // UE8M0 specifies the scale encoding, not a universal block geometry.
  // Activation and matrix weight grids are supplied by the model's numeric
  // contract; the default remains 1x32.
  if (format == MxFp8E4M3 || format == MxFp8E5M2) &&
    !(grouping is Block(rows=_, columns=_)) {
    raise InvalidGrouping
  }
  if scale_element != Absent && grouping == Absent {
    raise InvalidGrouping
  }
  let (scale_rows, scale_columns) = match grouping {
    Absent => (0, 0)
    Tensor => (1, 1)
    Row => (rows, 1)
    Block(rows=br, columns=bc) => {
      if br <= 0 || bc <= 0 {
        raise InvalidGrouping
      }
      (1 + (rows - 1) / br, 1 + (columns - 1) / bc)
    }
  }
  let element = format.element()
  let row_bytes = (columns.to_int64() * element.bits().to_int64() + 7L) / 8L
  if rows.to_int64() > 9223372036854775807L / row_bytes {
    raise SizeOverflow
  }
  let payload_bytes = rows.to_int64() * row_bytes
  let cells = scale_rows.to_int64() * scale_columns.to_int64()
  if scale_element.bytes() > 0 &&
    cells > 9223372036854775807L / scale_element.bytes().to_int64() {
    raise SizeOverflow
  }
  {
    format,
    element,
    rows,
    columns,
    grouping,
    scale_element,
    global_scale,
    row_bytes,
    payload_bytes,
    scale_rows,
    scale_columns,
    scale_bytes: cells * scale_element.bytes().to_int64(),
  }
}

///|
pub fn Representation::format(self : Representation) -> Format {
  self.format
}

///|
pub fn Representation::element(self : Representation) -> Element {
  self.element
}

///|
pub fn Representation::shape(self : Representation) -> (Int, Int) {
  (self.rows, self.columns)
}

///|
pub fn Representation::grouping(self : Representation) -> ScaleGrouping {
  self.grouping
}

///|
pub fn Representation::scale_element(self : Representation) -> ScaleElement {
  self.scale_element
}

///|
pub fn Representation::has_global_scale(self : Representation) -> Bool {
  self.global_scale
}

///|
pub fn Representation::row_bytes(self : Representation) -> Int64 {
  self.row_bytes
}

///|
pub fn Representation::payload_bytes(self : Representation) -> Int64 {
  self.payload_bytes
}

///|
pub fn Representation::scale_shape(self : Representation) -> (Int, Int) {
  (self.scale_rows, self.scale_columns)
}

///|
pub fn Representation::scale_bytes(self : Representation) -> Int64 {
  self.scale_bytes
}

///|
pub fn Representation::total_bytes(
  self : Representation,
) -> Int64 raise PrecisionError {
  let global = if self.global_scale { 4L } else { 0L }
  if self.scale_bytes > 9223372036854775807L - global {
    raise SizeOverflow
  }
  let extra = self.scale_bytes + global
  if self.payload_bytes > 9223372036854775807L - extra {
    raise SizeOverflow
  }
  self.payload_bytes + extra
}

///|
pub fn Representation::scale_index(
  self : Representation,
  row : Int,
  column : Int,
) -> Int64 raise PrecisionError {
  if row < 0 || row >= self.rows || column < 0 || column >= self.columns {
    raise IndexOutOfBounds
  }
  match self.grouping {
    Absent => raise InvalidGrouping
    Tensor => 0L
    Row => row.to_int64()
    Block(rows=br, columns=bc) =>
      (row / br).to_int64() * self.scale_columns.to_int64() +
      (column / bc).to_int64()
  }
}

///|
/// Stable semantics identity includes packing, scale grid and round boundaries.
pub fn Representation::canonical(self : Representation) -> String {
  let grouping = match self.grouping {
    Absent => "none"
    Tensor => "tensor"
    Row => "row"
    Block(rows=br, columns=bc) => "block:\{br}:\{bc}"
  }
  let scale = match self.scale_element {
    Absent => "none"
    F32 => "f32"
    Ue8M0 => "ue8m0"
    E4M3 => "e4m3fn"
  }
  "lunaflux.precision-representation.v1|\{self.format.name()}|\{self.rows}|\{self.columns}|\{grouping}|\{scale}|global:\{self.global_scale}|row-low-nibble-first-zero-tail|rne-satfinite|reconstruct-mul-f32-then-global-f32"
}

///|
pub fn Representation::digest(self : Representation) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(self.canonical())))
}
