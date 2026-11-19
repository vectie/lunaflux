///|
/// Allocation payload only; module/driver/queue overhead is additional.
pub(all) struct Budget {
  text : @runtime.Budget
  vision_weight_bytes : Int64
  vision_storage_bytes : Int64
  image_metadata_bytes : Int64
  mutable_bytes : Int64
  total_allocation_bytes : Int64
}

///|
pub fn Budget::new(
  program : @multimodal.Program,
  vision : @vision.Program,
  text_weight_bytes~ : Int64,
  vision_weight_bytes~ : Int64,
) -> Self raise {
  let text = @runtime.Budget::for_text(
    program.text_program(),
    weight_bytes=text_weight_bytes,
  )
  let vision_storage_bytes = vision.storage_bytes()
  let image_metadata_bytes = program.metadata_bytes()
  let extra = vision_storage_bytes + image_metadata_bytes
  guard vision_weight_bytes >= 0L &&
    vision_weight_bytes <= 9223372036854775807L - extra &&
    text.total_allocation_bytes <=
    9223372036854775807L - extra - vision_weight_bytes else {
    fail("invalid combined resident allocation budget")
  }
  {
    text,
    vision_weight_bytes,
    vision_storage_bytes,
    image_metadata_bytes,
    mutable_bytes: text.storage_bytes + text.output_bytes + extra,
    total_allocation_bytes: text.total_allocation_bytes +
    extra +
    vision_weight_bytes,
  }
}
