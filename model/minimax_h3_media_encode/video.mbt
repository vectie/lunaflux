///|
pub struct VideoEncoder {
  graph : @plan.Graph
  frames : Int
  height : Int
  width : Int
  latent_time : Int
  noise_value : Int
  means_value : Int
  deviations_value : Int
  tile_count : Int
}

///|
fn Builder::conv3d(
  self : Self,
  input : Int,
  name : String,
  ci : Int,
  co : Int,
  t : Int,
  h : Int,
  w : Int,
  kernel : Int,
  st : Int,
  ss : Int,
  before : Int,
  after : Int,
) -> (Int, Int, Int, Int) raise @plan.PlanError {
  let op : @plan.Operation = Conv3d(
    input_channels=ci,
    output_channels=co,
    frames=t,
    height=h,
    width=w,
    kernel~,
    temporal_stride=st,
    spatial_stride=ss,
    spatial_before=before,
    spatial_after=after,
  )
  let (ot, oh, ow) = @plan.conv3d_shape(t, h, w, kernel, st, ss, before, after)
  let output = self.emit(op, [
    input,
    self.weight(name + ".weight", [
      co.to_int64(),
      ci.to_int64(),
      kernel.to_int64(),
      kernel.to_int64(),
      kernel.to_int64(),
    ]),
    self.weight(name + ".bias", [co.to_int64()]),
  ])
  (output, ot, oh, ow)
}

///|
fn Builder::group_silu(
  self : Self,
  input : Int,
  name : String,
  c : Int,
  t : Int,
  h : Int,
  w : Int,
) -> Int raise @plan.PlanError {
  self.emit(GroupNormSilu(channels=c, frames=t, height=h, width=w, groups=32), [
    input,
    self.weight(name + ".weight", [c.to_int64()]),
    self.weight(name + ".bias", [c.to_int64()]),
  ])
}

///|
fn Builder::encode_tile(
  self : Self,
  input : Int,
  frames : Int,
  height : Int,
  width : Int,
) -> Int raise @plan.PlanError {
  let (first, _, _, _) = self.conv3d(
    input, "encoder.conv_in", 3, 128, frames, height, width, 3, 1, 1, 1, 1,
  )
  let mut state = first
  let mut ci = 128
  let mut t = frames
  let mut h = height
  let mut w = width
  for stage, co in [128, 256, 256, 512, 512, 1024] {
    for unit in 0..<2 {
      let prefix = "encoder.down_blocks.\{stage}.resnets.\{unit}"
      let norm = self.group_silu(state, prefix + ".norm1", ci, t, h, w)
      let (conv, _, _, _) = self.conv3d(
        norm,
        prefix + ".conv1",
        ci,
        co,
        t,
        h,
        w,
        3,
        1,
        1,
        1,
        1,
      )
      let norm = self.group_silu(conv, prefix + ".norm2", co, t, h, w)
      let (branch, _, _, _) = self.conv3d(
        norm,
        prefix + ".conv2",
        co,
        co,
        t,
        h,
        w,
        3,
        1,
        1,
        1,
        1,
      )
      let residual = if ci != co {
        let (projected, _, _, _) = self.conv3d(
          state,
          prefix + ".conv_shortcut",
          ci,
          co,
          t,
          h,
          w,
          1,
          1,
          1,
          0,
          0,
        )
        projected
      } else {
        state
      }
      state = self.emit(
        Add(elements=co.to_int64() * t.to_int64() * h.to_int64() * w.to_int64()),
        [branch, residual],
      )
      ci = co
    }
    if stage < 4 {
      let (next, nt, nh, nw) = self.conv3d(
        state,
        "encoder.down_blocks.\{stage}.downsamplers.0.conv",
        co,
        co,
        t,
        h,
        w,
        3,
        if stage == 1 || stage == 2 {
          2
        } else {
          1
        },
        2,
        0,
        1,
      )
      state = next
      t = nt
      h = nh
      w = nw
    }
  }
  let norm = self.group_silu(state, "encoder.norm_out", 1024, t, h, w)
  let (moments, _, _, _) = self.conv3d(
    norm, "encoder.conv_out", 1024, 48, t, h, w, 3, 1, 1, 1, 1,
  )
  let (quantized, _, _, _) = self.conv3d(
    moments, "quant_conv", 48, 48, t, h, w, 1, 1, 1, 0, 0,
  )
  quantized
}

///|
/// Complete reference image/video encoder graph. Input is ImageNet-normalized
/// F32 [3,T,H,W]. Explicit standard-normal noise is [24,latentT,H/16,W/16].
/// Complete spatial tiles blend moments (vertical then horizontal), temporal
/// clips replicate the last frame, then token_drop=3 precedes posterior sample.
pub fn VideoEncoder::new(
  frames~ : Int,
  height~ : Int,
  width~ : Int,
  image~ : Bool,
  max_tiles~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise @plan.PlanError {
  guard frames > 0 && frames <= 65536 && (!image || frames == 1) else {
    raise @plan.PlanError::InvalidGeometry
  }
  let ys = TileAxis::new(height)
  let xs = TileAxis::new(width)
  let slices = if image { 1 } else { (frames + 16) / 17 }
  guard max_tiles > 0 &&
    slices <= max_tiles / ys.starts.length() / xs.starts.length() &&
    3L * frames.to_int64() * height.to_int64() * width.to_int64() <= 2147483647L else {
    raise @plan.PlanError::BudgetExceeded
  }
  let chunk_frames = if image { 1 } else { 17 }
  let chunk_latent = if image { 1 } else { 5 }
  let latent_time = if image { 1 } else { slices * 5 - 3 }
  let lh = height / 16
  let lw = width / 16
  let b : Builder = {
    video: true,
    weight_ids: Map([]),
    values: [
      {
        elements: 3L * frames.to_int64() * height.to_int64() * width.to_int64(),
        origin: Input,
      },
    ],
    nodes: [],
  }
  let mut all_moments = b.emit(
    Zeros(elements=48L * latent_time.to_int64() * lh.to_int64() * lw.to_int64()),
    [],
  )
  let th = ys.extent / 16
  let tw = xs.extent / 16
  for clip in 0..<slices {
    let tiles : Array[Int] = []
    for y in ys.starts {
      for x in xs.starts {
        let input = b.emit(
          VideoSlice(
            frames~,
            height~,
            width~,
            start_frame=clip * 17,
            start_y=y,
            start_x=x,
            tile_frames=chunk_frames,
            tile_height=ys.extent,
            tile_width=xs.extent,
          ),
          [0],
        )
        tiles.push(b.encode_tile(input, chunk_frames, ys.extent, xs.extent))
      }
    }
    let mut canvas = b.emit(
      Zeros(
        elements=48L * chunk_latent.to_int64() * lh.to_int64() * lw.to_int64(),
      ),
      [],
    )
    for yi, y in ys.starts {
      for xi, x in xs.starts {
        let id = yi * xs.starts.length() + xi
        let tile = tiles[id]
        let above = if yi > 0 { tiles[id - xs.starts.length()] } else { tile }
        let left = if xi > 0 { tiles[id - 1] } else { tile }
        let copy_height = th -
          (if yi < ys.overlaps.length() { ys.overlaps[yi] / 16 } else { 0 })
        let copy_width = tw -
          (if xi < xs.overlaps.length() { xs.overlaps[xi] / 16 } else { 0 })
        canvas = b.emit(
          InsertTile(
            channels=48,
            frames=chunk_latent,
            height=lh,
            width=lw,
            tile_height=th,
            tile_width=tw,
            y=y / 16,
            x=x / 16,
            copy_height~,
            copy_width~,
            overlap_y=if yi > 0 { ys.overlaps[yi - 1] / 16 } else { 0 },
            overlap_x=if xi > 0 { xs.overlaps[xi - 1] / 16 } else { 0 },
          ),
          [canvas, tile, above, left],
        )
      }
    }
    all_moments = b.emit(
      InsertTime(
        channels=48,
        frames=latent_time,
        height=lh,
        width=lw,
        tile_frames=chunk_latent,
        start=clip * chunk_latent,
        count=Int::min(chunk_latent, latent_time - clip * chunk_latent),
      ),
      [all_moments, canvas],
    )
  }
  let noise_value = b.values.length()
  b.values.push({
    elements: 24L * latent_time.to_int64() * lh.to_int64() * lw.to_int64(),
    origin: Input,
  })
  let means_value = b.values.length()
  b.values.push({ elements: 24L, origin: Input, })
  let deviations_value = b.values.length()
  b.values.push({ elements: 24L, origin: Input, })
  let output = b.emit(
    GaussianPatch(channels=24, frames=latent_time, height=lh, width=lw),
    [all_moments, noise_value, means_value, deviations_value],
  )
  {
    graph: @plan.Graph::new(b.values, b.nodes, output~, max_scratch_bytes~),
    frames,
    height,
    width,
    latent_time,
    noise_value,
    means_value,
    deviations_value,
    tile_count: slices * ys.starts.length() * xs.starts.length(),
  }
}
