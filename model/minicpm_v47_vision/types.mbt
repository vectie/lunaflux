///|
/// Concatenated full-attention crop boundaries, expanded to one start/end per
/// query. They are request-preparation buffers, not decoder KV pages.
pub struct Spans {
  priv starts : ReadOnlyArray[Int32]
  priv ends : ReadOnlyArray[Int32]
  priv boundaries : ReadOnlyArray[Int32]
}

///|
pub fn Spans::starts(self : Self) -> ReadOnlyArray[Int32] {
  self.starts
}

///|
pub fn Spans::ends(self : Self) -> ReadOnlyArray[Int32] {
  self.ends
}

///|
pub fn Spans::boundaries(self : Self) -> ReadOnlyArray[Int32] {
  self.boundaries
}

///|
pub fn Spans::rows(self : Self) -> Int32 {
  self.starts.length()
}

///|
/// Prepared packed-crop geometry. Original pixel patches are raster ordered.
/// Early windows reorder four rows before attention/flattening, and the final
/// merger separately reorders four post-encoder rows before its MLP.
pub struct Layout {
  priv grids : ReadOnlyArray[@positions.Grid]
  priv raw : Spans
  priv encoded : Spans
  priv windows : Spans
  priv position_ids : ReadOnlyArray[Int32]
  priv window_indices : ReadOnlyArray[Int32]
  priv merger_indices : ReadOnlyArray[Int32]
  priv output_boundaries : ReadOnlyArray[Int32]
  priv insert_after : Int32?
}

///|
pub fn Layout::grids(self : Self) -> ReadOnlyArray[@positions.Grid] {
  self.grids
}

///|
pub fn Layout::raw_spans(self : Self) -> Spans {
  self.raw
}

///|
pub fn Layout::encoded_spans(self : Self) -> Spans {
  self.encoded
}

///|
pub fn Layout::window_spans(self : Self) -> Spans {
  self.windows
}

///|
pub fn Layout::position_ids(self : Self) -> ReadOnlyArray[Int32] {
  self.position_ids
}

///|
pub fn Layout::window_indices(self : Self) -> ReadOnlyArray[Int32] {
  self.window_indices
}

///|
pub fn Layout::merger_indices(self : Self) -> ReadOnlyArray[Int32] {
  self.merger_indices
}

///|
pub fn Layout::output_boundaries(self : Self) -> ReadOnlyArray[Int32] {
  self.output_boundaries
}

///|
pub fn Layout::insert_after(self : Self) -> Int32? {
  self.insert_after
}

///|
pub fn Layout::output_rows(self : Self) -> Int32 {
  self.output_boundaries[self.output_boundaries.length() - 1]
}

///|
pub fn Layout::layer_rows(self : Self, layer : Int32) -> Int32 raise {
  guard layer >= 0 && layer < 27 else {
    fail("vision layer outside selected encoder")
  }
  if self.insert_after is Some(index) && layer > index {
    self.encoded.rows()
  } else {
    self.raw.rows()
  }
}

///|
/// Model-owned admission for generic attention, preserving head width72.
pub fn attention_plan(
  spec : @model.Spec,
  maximum_patches~ : Int32,
  maximum_crop_patches~ : Int32,
) -> @precision.SegmentedAttentionPrecision raise {
  let vision = spec.vision()
  @precision.SegmentedAttentionPrecision::new(
    maximum_rows=maximum_patches,
    maximum_segment=maximum_crop_patches,
    head_count=vision.heads,
    head_width=vision.hidden_width / vision.heads,
  )
}
