///|
pub fn JointPackedModalitySelection::new(
  request : JointDiffusionIoPlan,
  packed_timestep : JointPackedTimestepSelection,
  video_indices~ : ArrayView[Int],
  audio_indices~ : ArrayView[Int],
) -> Self raise JointDiffusionIoPlanError {
  if packed_timestep.identity() != request.identity() ||
    packed_timestep.request_digest() != request.request_digest() {
    raise InvalidPackedModalitySelection
  }
  let shape = request.shape()
  let expected_video = shape.video_latent_frames() /
    request.video_patch_temporal() *
    (shape.video_latent_height() / request.video_patch_height()) *
    (shape.video_latent_width() / request.video_patch_width())
  let expected_audio = shape.audio_token_rows()
  if video_indices.length() != expected_video ||
    audio_indices.length() != expected_audio {
    raise InvalidPackedModalitySelection
  }
  let packed_rows = packed_timestep.packed_row_count()
  let seen = Array::make(packed_rows, false)
  let video : Array[Int] = []
  let audio : Array[Int] = []
  for index in video_indices {
    if index < 0 || index >= packed_rows || seen[index] {
      raise InvalidPackedModalitySelection
    }
    seen[index] = true
    video.push(index)
  }
  for index in audio_indices {
    if index < 0 || index >= packed_rows || seen[index] {
      raise InvalidPackedModalitySelection
    }
    seen[index] = true
    audio.push(index)
  }
  let text = StringBuilder()
  text <+
    "lunaflux.joint-packed-modality-selection.v1|model=\{request.identity().content().as_hex()}:\{request.identity().plan().as_hex()}|request=\{request.request_digest().as_hex()}|timestep=\{packed_timestep.digest().as_hex()}|rows=\{packed_rows}|video="
  for offset, index in video {
    if offset > 0 {
      text.write_char(',')
    }
    text.write_string(index.to_string())
  }
  text.write_string("|audio=")
  for offset, index in audio {
    if offset > 0 {
      text.write_char(',')
    }
    text.write_string(index.to_string())
  }
  let digest = try! @spec.PlanDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(text.to_string()))),
  )
  {
    identity: request.identity(),
    request_digest: request.request_digest(),
    packed_timestep_digest: packed_timestep.digest(),
    digest,
    packed_row_count: packed_rows,
    video_indices: ReadOnlyArray::from_array(video),
    audio_indices: ReadOnlyArray::from_array(audio),
  }
}

///|
pub fn JointPackedModalitySelection::digest(self : Self) -> @spec.PlanDigest {
  self.digest
}

///|
pub fn JointPackedModalitySelection::identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn JointPackedModalitySelection::request_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.request_digest
}

///|
pub fn JointPackedModalitySelection::packed_timestep_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.packed_timestep_digest
}

///|
pub fn JointPackedModalitySelection::packed_row_count(self : Self) -> Int {
  self.packed_row_count
}

///|
pub fn JointPackedModalitySelection::video_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.video_indices
}

///|
pub fn JointPackedModalitySelection::audio_indices(
  self : Self,
) -> ReadOnlyArray[Int] {
  self.audio_indices
}
