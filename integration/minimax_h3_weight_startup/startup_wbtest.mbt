///|
test "startup preflight keeps typed component boundaries and bounded accounting" {
  let vision = Packing::VisionConditioner
  let text = Packing::TextEncoder
  let total = vision.bytes() + text.bytes()
  assert_eq(component_bytes(TextConditioner, [text, vision][:], total), total)
  try component_bytes(TextConditioner, [text, vision][:], total - 1L) catch {
    BudgetExceeded => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("budget accepted")
  }
  try component_bytes(VideoVae, [vision][:], total) catch {
    InvalidPacking => ()
    _ => fail("wrong failure")
  } noraise {
    _ => fail("wrong component accepted")
  }
  assert_eq(
    component_bytes(TextConditioner, [vision][:], 9223372036854775807L),
    vision.bytes(),
  )
}

///|
test "startup close is deterministic and forbids reopening" {
  let owner = WeightStartup::new(max_device_bytes=1L)
  assert_eq(owner.reserved_bytes(), 0L)
  assert_eq(owner.count(), 0)
  try owner.acquire() catch {
    _ => ()
  } noraise {
    _ => fail("empty owner leased")
  }
  owner.close()
  owner.close()
  assert_true(owner.closed && owner.failed)
}

///|
test "packing lists retain caller specified block and head order" {
  let a = @weights.video_block_weight_packing(layer=0, max_bytes=1000000000L)
  let b = @weights.video_block_weight_packing(layer=1, max_bytes=1000000000L)
  let h = @weights.video_head_weight_packing(max_bytes=1000000000L)
  let items = [Packing::VideoBlock(a), VideoBlock(b), VideoFrontend(h)]
  assert_eq(
    component_bytes(VideoVae, items[:], 1000000000L),
    a.byte_count + b.byte_count + h.byte_count,
  )
  assert_eq(items[0].bytes(), a.byte_count)
  assert_eq(items[2].bytes(), h.byte_count)
}

///|
test "lease release is idempotent and blocks close until released" {
  let owner = WeightStartup::new(max_device_bytes=1L)
  owner.leases = 1
  let lease : WeightLease = { owner, released: false, }
  try owner.close() catch {
    _ => ()
  } noraise {
    _ => fail("live lease accepted")
  }
  lease.release()
  lease.release()
  assert_eq(owner.leases, 0)
  try lease.entry(0) catch {
    _ => ()
  } noraise {
    _ => fail("released lease accepted")
  }
  owner.close()
}

///|
test "official decoder factories cover every packed group in execution order" {
  let maximum = 100000000000L
  let video = video_decoder_packings(max_bytes=maximum)
  assert_eq(video.length(), 38)
  assert_true(video[0] is VideoFrontend(_))
  for i in 0..<36 {
    guard video[i + 1] is VideoBlock(packing) else { fail("missing block") }
    assert_eq(packing.layer, i)
  }
  assert_true(video[37] is VideoFrontend(_))
  let audio = audio_decoder_packings(max_bytes=maximum)
  assert_eq(audio.length(), 16)
  guard audio[0] is AudioDecoder(front) else { fail("missing frontend") }
  assert_true(front.group is Frontend)
  for i in 0..<7 {
    guard audio[1 + i * 2] is AudioDecoder(up) else {
      fail("missing upsampler")
    }
    assert_true(up.group == Upsample(i))
    assert_true(audio[2 + i * 2] is AudioAmp(_))
  }
  guard audio[15] is AudioDecoder(tail) else { fail("missing tail") }
  assert_true(tail.group is WaveformTail)
}

///|
test "reference segmented reservation bounds every alignment gap without overflow" {
  assert_eq(reference_reservation(1024L, 3, 1213L), 1213L)
  try reference_reservation(1024L, 3, 1212L) catch {
    _ => ()
  } noraise {
    _ => fail("padding budget accepted")
  }
  try
    reference_reservation(9223372036854775800L, 1, 9223372036854775807L)
  catch {
    _ => ()
  } noraise {
    _ => fail("overflow accepted")
  }
  try reference_reservation(0L, 1, 100L) catch {
    _ => ()
  } noraise {
    _ => fail("empty payload accepted")
  }
}
