///|
/// Pixel dimensions, rather than a patch grid.
pub(all) struct Size {
  height : Int32
  width : Int32
} derive(Eq, Debug)

///|
/// A thumbnail or a row-major crop of the resized detail canvas.
pub(all) struct Crop {
  canvas : Size
  top : Int32
  left : Int32
  size : Size
}

///|
/// Immutable image preparation geometry. This owner has no camera, codec,
/// tokenizer, device, or robot dependency.
pub struct Plan {
  priv input : Size
  priv grid : (Int32, Int32)
  priv crops : ReadOnlyArray[Crop]
  priv markers : @model.Markers
  priv merge : Int32
  priv columns : Int32
  priv token_count : Int32
}

///|
pub fn Plan::input_size(self : Self) -> Size {
  self.input
}

///|
/// (Rows, columns), or (0, 0) when there are no detail slices.
pub fn Plan::slice_grid(self : Self) -> (Int32, Int32) {
  self.grid
}

///|
pub fn Plan::crops(self : Self) -> ReadOnlyArray[Crop] {
  self.crops
}

///|
pub fn Plan::target_sizes(self : Self) -> ReadOnlyArray[@positions.Grid] {
  self.crops.map(crop => {
    height: crop.size.height / 14,
    width: crop.size.width / 14,
  })
}

///|
/// Number of visual embeddings after compression; excludes framing tokens.
pub fn Plan::visual_tokens(self : Self) -> Int32 {
  self.token_count
}

///|
/// Final dimension of upstream [1, 3, 14, columns] pixel_values.
pub fn Plan::pixel_columns(self : Self) -> Int32 {
  self.columns
}

///|
pub fn Plan::pixel_bytes(self : Self) -> Int64 {
  self.columns.to_int64() * 3L * 14L * 2L
}

///|
pub struct Placeholder {
  priv tokens : ReadOnlyArray[Int32]
  priv modalities : ReadOnlyArray[@positions.Modality]
  priv grids : ReadOnlyArray[@positions.Grid]
}

///|
pub fn Placeholder::tokens(self : Self) -> ReadOnlyArray[Int32] {
  self.tokens
}

///|
pub fn Placeholder::modalities(
  self : Self,
) -> ReadOnlyArray[@positions.Modality] {
  self.modalities
}

///|
pub fn Placeholder::grids(self : Self) -> ReadOnlyArray[@positions.Grid] {
  self.grids
}

///|
/// BF16 little-endian pixels in [1,3,14,Plan::pixel_columns()] order.
/// Crop/channel/patch-row ordering matches concatenation along the last axis.
pub struct Pixels {
  priv plan : Plan
  priv data : Bytes
}

///|
pub fn Pixels::plan(self : Self) -> Plan {
  self.plan
}

///|
pub fn Pixels::data(self : Self) -> Bytes {
  self.data
}
