///|
fn modality_fixture() -> (
  JointDiffusionIoPlan,
  JointPackedTimestepSelection,
  Array[Int],
  Array[Int],
) {
  let request = io_text_request(io_minimax_plan())
  let video_count = 37296
  let audio_count = 414
  let packed_count = video_count + audio_count + 137
  let timestep_indices = Array::make(packed_count, 0)
  let timestep = try! JointPackedTimestepSelection::new(
    request,
    evaluation_index=0,
    distinct_timestep_count=3,
    indices=timestep_indices,
  )
  let video = Array::makei(video_count, index => index)
  let audio = Array::makei(audio_count, index => video_count + index)
  (request, timestep, video, audio)
}

///|
test "packed modality selection binds exact target rows and ordered identity" {
  let (request, timestep, video, audio) = modality_fixture()
  let selection = JointPackedModalitySelection::new(
    request,
    timestep,
    video_indices=video,
    audio_indices=audio,
  )
  video[0] = 99999
  assert_eq(selection.video_indices().length(), 37296)
  assert_eq(selection.audio_indices().length(), 414)
  assert_eq(selection.video_indices()[0], 0)
  assert_eq(selection.audio_indices()[0], 37296)
  assert_eq(selection.packed_row_count(), 37847)
  let (_, _, repeated_video, repeated_audio) = modality_fixture()
  let repeated = JointPackedModalitySelection::new(
    request,
    timestep,
    video_indices=repeated_video,
    audio_indices=repeated_audio,
  )
  assert_eq(selection.digest(), repeated.digest())
}

///|
test "packed modality selection rejects wrong counts range and global duplicates" {
  let (request, timestep, video, audio) = modality_fixture()
  let short_video = video.copy()
  let _ = short_video.pop()
  let out_of_range = audio.copy()
  out_of_range[0] = timestep.packed_row_count()
  let duplicate = audio.copy()
  duplicate[0] = video[0]
  for
    hostile in [(short_video, audio), (video, out_of_range), (video, duplicate)] {
    let (hostile_video, hostile_audio) = hostile
    try
      JointPackedModalitySelection::new(
        request,
        timestep,
        video_indices=hostile_video,
        audio_indices=hostile_audio,
      )
    catch {
      InvalidPackedModalitySelection => ()
      _ => fail("unexpected packed modality rejection")
    } noraise {
      _ => fail("hostile packed modality selection admitted")
    }
  }
}
