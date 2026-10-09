///|
// The native test double has process-wide resource and wait counters. Async
// socket tests yield while resources are live; serialize the device portions,
// not checkpoint file construction, so another fixture cannot alter a sample.
let device_fixture_lane : @async.Mutex = @async.Mutex()
