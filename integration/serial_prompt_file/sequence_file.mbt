///|
// Metadata parsing is CPU preparation only. This plan is independent of the
// model family and contains no device effects or payload authentication.
fn sequence_frame_counts(
  bytes : Bytes,
  requests : Int,
  maximum_context : Int,
) -> Array[Int] raise {
  guard requests > 0 &&
    requests <= maximum_context &&
    bytes.length().to_int64() <= requests.to_int64() * 12L else {
    fail("invalid request-sequence metadata envelope")
  }
  let result : Array[Int] = []
  for field in @utf8.decode(bytes).trim().split(",") {
    guard result.length() < requests else {
      fail("request-sequence metadata has extra entries")
    }
    let count = @string.parse_int(field.trim())
    guard count > 0 && count <= maximum_context else {
      fail("invalid request frame count")
    }
    result.push(count)
  }
  guard result.length() == requests else {
    fail("request-sequence metadata is incomplete")
  }
  result
}

///|
/// Read frame-counts.txt and prepare request-i/ under the same bounded file
/// adapter. All requests are checked before device startup; neither metadata
/// nor frame payloads are reopened, hashed or retained through file authority.
pub fn load_generation_sequence_file(
  root : @approved_fs.ApprovedRoot,
  requests : Int,
  limits : @protocol.WorkerProtocolLimits,
  inference : @inference.InferenceLimits,
  maximum_new_tokens~ : Int,
  maximum_context~ : Int,
  maximum_bytes~ : Int64,
  model_generation~ : UInt64,
) -> ReadOnlyArray[@prompt.PreparedSerialPrompt] raise {
  guard requests > 0 && requests <= maximum_context else {
    fail("invalid bounded request count")
  }
  let metadata = {
    let file = root.open_file(
      @approved_fs.ApprovedRelativeLocator::new("frame-counts.txt"),
    )
    defer file.close()
    file.read_immutable_snapshot(
      maximum_bytes=(requests.to_int64() * 12L).min(maximum_bytes),
    )
  }
  load_generation_sequence(
    root,
    sequence_frame_counts(metadata, requests, maximum_context),
    limits,
    inference,
    maximum_new_tokens~,
    maximum_context~,
    maximum_bytes~,
    model_generation~,
  )
}
