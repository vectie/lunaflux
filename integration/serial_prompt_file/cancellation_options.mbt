///|
/// Optional CLI replay control, parsed once before checkpoint/device startup.
/// The runtime owns cancellation effects; this adapter only prepares the pure
/// request/step decision and strips its explicit suffix from ordinary arguments.
pub fn split_request_cancellation_options(
  arguments : ArrayView[String],
) -> (ArrayView[String], @generation.SerialRequestCancellation?) raise {
  guard arguments
    is [.. rest, "--cancel-request", request, "--after-steps", steps] else {
    guard !arguments.contains("--cancel-request") &&
      !arguments.contains("--after-steps") else {
      fail("cancellation requires --cancel-request INDEX --after-steps COUNT")
    }
    return (arguments, None)
  }
  guard !rest.contains("--cancel-request") && !rest.contains("--after-steps") else {
    fail("duplicate request cancellation control")
  }
  (
    rest,
    Some(
      @generation.SerialRequestCancellation::new(
        request_index=@string.parse_int(request),
        after_submitted_steps=@string.parse_int(steps),
      ),
    ),
  )
}
