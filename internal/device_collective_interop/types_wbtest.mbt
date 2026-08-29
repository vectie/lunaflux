///|
test "collective interop tokens are distinct opaque nominal capabilities" {
  let _ : (ContextToken) -> Unit = fn(_context) { () }
  let _ : (RegionToken) -> Unit = fn(_region) { () }
  let _ : (QueueToken) -> Unit = fn(_queue) { () }
}
