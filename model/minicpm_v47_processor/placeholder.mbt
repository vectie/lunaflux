///|
/// Image body only. Ingress may prepend tokenized <image_id>N</image_id> as
/// ordinary Text tokens; IDs belong to each prompt, not to the pixel plan.
pub fn Plan::placeholder(self : Self) -> Placeholder {
  let tokens = []
  let modalities = []
  for index in 0..<self.crops.length() {
    let crop = self.crops[index]
    if index > 1 && (index - 1) % self.grid.1 == 0 {
      tokens.push(self.markers.newline)
      modalities.push(@positions.Modality::Text)
    }
    tokens.push(
      if index == 0 {
        self.markers.image_start
      } else {
        self.markers.slice_start
      },
    )
    modalities.push(Text)
    let count = crop.size.height /
      (14 * self.merge) *
      (crop.size.width / (14 * self.merge))
    for _ in 0..<count {
      tokens.push(self.markers.image)
      modalities.push(Image)
    }
    tokens.push(
      if index == 0 {
        self.markers.image_end
      } else {
        self.markers.slice_end
      },
    )
    modalities.push(Text)
  }
  {
    tokens: ReadOnlyArray::from_array(tokens),
    modalities: ReadOnlyArray::from_array(modalities),
    grids: self.target_sizes(),
  }
}

///|
/// Upstream replacement text, including a local image ID when supplied.
pub fn Plan::placeholder_text(self : Self, image_id? : Int32) -> String raise {
  let builder = StringBuilder()
  if image_id is Some(id) {
    guard id >= 0 else { fail("negative local image ID") }
    builder.write_string("<image_id>\{id}</image_id>")
  }
  for index in 0..<self.crops.length() {
    if index > 1 && (index - 1) % self.grid.1 == 0 {
      builder.write_string("\n")
    }
    builder.write_string(if index == 0 { "<image>" } else { "<slice>" })
    let crop = self.crops[index]
    let count = crop.size.height /
      (14 * self.merge) *
      (crop.size.width / (14 * self.merge))
    for _ in 0..<count {
      builder.write_string("<|image_pad|>")
    }
    builder.write_string(if index == 0 { "</image>" } else { "</slice>" })
  }
  builder.to_string()
}
