///|
priv struct Case {
  name : String
  downsample : @model.Downsample
  tokens : Array[Int32]
  modalities : Array[@positions.Modality]
  images : Array[@positions.Grid]
  videos : Array[@positions.Grid]
}

///|
fn Case::new(name : String, mode : @model.Downsample) -> Self {
  {
    name,
    downsample: mode,
    tokens: [],
    modalities: [],
    images: [],
    videos: [],
  }
}

///|
fn Case::text(self : Self, tokens : ArrayView[Int32]) -> Unit {
  for token in tokens {
    self.tokens.push(token)
    self.modalities.push(Text)
  }
}

///|
fn Case::crop(
  self : Self,
  spec : @model.Spec,
  grid : @positions.Grid,
  thumbnail~ : Bool,
  video~ : Bool = false,
) -> Unit {
  let m = spec.markers()
  self.text([if thumbnail { m.image_start } else { m.slice_start }])
  let merge = match self.downsample {
    FourX => 2
    SixteenX => 4
  }
  for _ in 0..<(grid.height / merge * (grid.width / merge)) {
    self.tokens.push(if video { m.video } else { m.image })
    self.modalities.push(if video { Video } else { Image })
  }
  self.text([if thumbnail { m.image_end } else { m.slice_end }])
  if video {
    self.videos.push(grid)
  } else {
    self.images.push(grid)
  }
}

///|
fn Case::frame(
  self : Self,
  spec : @model.Spec,
  height : Int32,
  width : Int32,
  video~ : Bool = false,
) -> Unit raise {
  let fragment = @processor.Plan::build(spec, height~, width~).placeholder()
  let m = spec.markers()
  for i, token in fragment.tokens() {
    self.tokens.push(if video && token == m.image { m.video } else { token })
    self.modalities.push(
      if video && fragment.modalities()[i] is Image {
        Video
      } else {
        fragment.modalities()[i]
      },
    )
  }
  for grid in fragment.grids() {
    if video {
      self.videos.push(grid)
    } else {
      self.images.push(grid)
    }
  }
}

///|
fn Case::json(self : Self) -> Json {
  let images = self.images.map(g => [g.height, g.width])
  let videos = self.videos.map(g => [g.height, g.width])
  let modalities = self.modalities.map(m => {
    match m {
      Text => 0
      Image => 1
      Video => 2
    }
  })
  {
    "name": Json(self.name),
    "tokens": Json([self.tokens]),
    "modalities": Json([modalities]),
    "image_grids": Json(images),
    "video_grids": Json(videos),
    "downsample": Json(
      match self.downsample {
        FourX => "4x"
        SixteenX => "16x"
      },
    ),
  }
}

///|
fn corpus(original : String) -> Array[Case] raise {
  let cases = []
  for mode in [@model.Downsample::FourX, SixteenX] {
    let name = match mode {
      FourX => "4x"
      SixteenX => "16x"
    }
    let spec = @model.Spec::parse(
      @utf8.encode(
        if mode is FourX {
          original.replace(old="\"16x\"", new="\"4x\"")
        } else {
          original
        },
      )[:],
    )
    for count in [1, 3, 17] {
      let case = Case::new("\{name}-text-\{count}", mode)
      case.text(Array::makei(count, i => i + 10))
      cases.push(case)
    }
    for
      (height, width) in [
        (224, 224),
        (448, 896),
        (896, 896),
        (768, 1280),
        (1001, 1803),
        (3584, 448),
      ] {
      for prefix in [0, 1, 7] {
        let single = Case::new(
          "\{name}-single-\{height}-\{width}-\{prefix}",
          mode,
        )
        single.text(Array::makei(prefix, i => i + 10))
        single.frame(spec, height, width)
        single.text([42, 43])
        cases.push(single)
        for gap in 0..<3 {
          let case = Case::new(
            "\{name}-mixed-\{height}-\{width}-\{prefix}-\{gap}",
            mode,
          )
          case.text(Array::makei(prefix, i => i + 10))
          case.frame(spec, height, width)
          case.text(
            match gap {
              0 => []
              1 => [spec.markers().newline]
              _ => [42, spec.markers().newline]
            },
          )
          case.frame(spec, width, height, video=true)
          case.text(
            match gap {
              0 => []
              1 => [spec.markers().newline]
              _ => [45]
            },
          )
          case.frame(spec, 224, 224)
          case.text([46])
          cases.push(case)
        }
      }
    }
    let simple = Case::new("\{name}-native-grid", mode)
    simple.text([10])
    simple.crop(spec, { height: 8, width: 12, }, thumbnail=true)
    simple.text([20])
    cases.push(simple)
    let slices = Case::new("\{name}-native-slices", mode)
    slices.crop(spec, { height: 4, width: 12, }, thumbnail=true)
    for _ in 0..<3 {
      slices.crop(spec, { height: 4, width: 8, }, thumbnail=false)
    }
    slices.text([20])
    cases.push(slices)
  }
  cases
}
