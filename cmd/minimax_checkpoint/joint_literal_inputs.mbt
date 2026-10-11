///|
/// Diagnostic-only literal initialization. The numerical request/latent plan
/// owns dimensions; filesystem input is read once before opening CUDA.
fn check_joint_literal_lengths(
  video_bytes : Int64,
  audio_bytes : Int64,
  video_actual : Int,
  audio_actual : Int,
) -> Unit raise {
  guard video_bytes > 0L &&
    audio_bytes > 0L &&
    video_actual.to_int64() == video_bytes &&
    audio_actual.to_int64() == audio_bytes else {
    fail("literal initial video/audio extents differ from the prepared request")
  }
}

///|
fn read_joint_literal_inputs(
  root : @approved_fs.ApprovedRoot,
  plan : @loop.LatentStoragePlan,
  names : (String, String)?,
) -> (Bytes, Bytes)? raise {
  match names {
    None => None
    Some((video_name, audio_name)) => {
      let video_bytes = plan.region(VideoBoundary).byte_count
      let audio_bytes = plan.region(AudioState).byte_count
      let video = read(root, video_name, video_bytes)
      let audio = read(root, audio_name, audio_bytes)
      check_joint_literal_lengths(
        video_bytes,
        audio_bytes,
        video.length(),
        audio.length(),
      )
      Some((video, audio))
    }
  }
}

///|
fn upload_joint_literal_inputs(
  storage : @latent.LatentStorage,
  video : Bytes,
  audio : Bytes,
) -> Unit raise {
  // Both inputs are already shape-checked. Copy raw canonical channel-major
  // video; the ordinary production patchifier still runs afterwards. Audio
  // is the original token-major initial row tensor, not a terminal VAE latent.
  for
    (buffer, bytes) in [
      (@loop.LatentBuffer::VideoBoundary, video),
      (AudioState, audio),
    ] {
    storage.with_region(buffer, (allocation, offset, count) => {
      allocation.copy_from_host(bytes, 0L, offset, count)
    })
  }
}

///|
test "literal initialization checks both complete inputs before any copy" {
  check_joint_literal_lengths(16L, 8L, 16, 8)
  for (video, audio) in [(15, 8), (16, 7), (17, 8), (16, 9)] {
    try check_joint_literal_lengths(16L, 8L, video, audio) catch {
      _ => ()
    } noraise {
      _ => fail("incomplete literal initialization admitted")
    }
  }
}
