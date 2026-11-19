///|
pub(all) enum Content {
  Text(String)
  Image(@rgb8.Image)
}

///|
/// Prepared on the host before request execution. Image prompt/layout/pixels
/// are derived together so their crop and feature ordering cannot diverge.
pub enum Prepared {
  Text(@prompt.PreparedTextPrompt)
  Image(
    @multimodal.ImagePrompt,
    @vision.Input,
    images~ : Int32,
    pixel_bytes~ : Int64
  )
}

///|
pub fn Prepared::prompt(self : Self) -> @prompt.PreparedTextPrompt {
  match self {
    Text(prompt) => prompt
    Image(prompt, _, ..) => prompt.prompt()
  }
}

///|
pub fn Prepared::vision_input(
  self : Self,
) -> (@multimodal.ImagePrompt, @vision.Input)? {
  match self {
    Text(_) => None
    Image(prompt, input, ..) => Some((prompt, input))
  }
}

///|
pub fn Prepared::image_count(self : Self) -> Int32 {
  match self {
    Text(_) => 0
    Image(_, _, images~, ..) => images
  }
}

///|
pub fn Prepared::pixel_bytes(self : Self) -> Int64 {
  match self {
    Text(_) => 0L
    Image(_, _, pixel_bytes~, ..) => pixel_bytes
  }
}

///|
pub fn Prepared::patch_rows(self : Self) -> Int32 {
  match self {
    Text(_) => 0
    Image(_, input, ..) => input.patch_rows()
  }
}

///|
pub fn Prepared::feature_rows(self : Self) -> Int32 {
  match self {
    Text(_) => 0
    Image(prompt, _, ..) => prompt.feature_rows()
  }
}

///|
/// Pure model preparation from decoded frames. No file/network/device access.
/// Admit all crop geometry, token and pixel budgets before resize/packing.
pub fn prepare(
  spec : @model.Spec,
  tokenizer : @tokenizer.TokenizerSpec,
  vision : @vision.Program,
  system : String,
  content : ArrayView[Content],
  maximum_tokens~ : Int32,
  maximum_prepared_pixel_bytes~ : Int64,
  maximum_image_working_bytes~ : Int64 = 268435456L,
  system_present~ : Bool = true,
) -> Prepared raise {
  guard !content.is_empty() &&
    content.length() <= 64 &&
    maximum_prepared_pixel_bytes >= 0L else {
    fail("invalid media content/budget")
  }
  let parts : Array[@multimodal.Content] = []
  let images : Array[(@processor.Plan, @rgb8.Image)] = []
  let grids : Array[@positions.Grid] = []
  let text : Array[String] = []
  let mut pixel_bytes = 0L
  for part in content {
    match part {
      Text(value) => {
        parts.push(Text(value))
        text.push(value)
      }
      Image(image) => {
        guard images.length() < 8 else {
          fail("too many images in media request")
        }
        let plan = @processor.Plan::build(
          spec,
          height=image.height(),
          width=image.width(),
        )
        guard plan.preparation_bytes() <= maximum_image_working_bytes else {
          fail("image resize exceeds working memory budget")
        }
        guard plan.pixel_bytes() <= maximum_prepared_pixel_bytes - pixel_bytes else {
          fail("prepared image payload exceeds queue budget")
        }
        pixel_bytes += plan.pixel_bytes()
        images.push((plan, image))
        parts.push(Image(plan))
        for grid in plan.target_sizes() {
          grids.push(grid)
        }
      }
    }
  }
  if images.is_empty() {
    return Text(
      @prompt.prepare_text_prompt(
        spec,
        tokenizer,
        system,
        text.join("\n"),
        maximum_tokens~,
        system_present~,
      ),
    )
  }
  let (raw, _, _) = vision.row_capacities()
  let layout = @layout.Layout::build(
    spec,
    grids[:],
    maximum_patches=raw,
    maximum_crop_patches=vision.maximum_crop_patches(),
  )
  let prompt = @multimodal.prepare_image_prompt(
    spec,
    tokenizer,
    system,
    parts[:],
    maximum_tokens~,
    system_present~,
  )
  let pixels = images.map(pair => {
    pair.0.prepare_rgb(pair.1.data()[:], max_bytes=maximum_image_working_bytes)
  })
  Image(
    prompt,
    @vision.Input::from_images(vision, layout, pixels[:]),
    images=images.length(),
    pixel_bytes~,
  )
}
