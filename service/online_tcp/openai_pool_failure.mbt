///|
fn luna_openai_first_failure(
  previous : (String, Error)?,
  stage : String,
  error : Error,
) -> (String, Error)? {
  match previous {
    Some(_) => previous
    None => Some((stage, error))
  }
}

///|
fn LunaOnlineOpenAIConnectionPool::fail_pool(
  self : LunaOnlineOpenAIConnectionPool,
  stage : String,
  error : Error,
) -> Unit {
  self.failure = luna_openai_first_failure(self.failure, stage, error)
  self.begin_drain()
}

///|
/// First terminal failure, retained across cleanup retries. Reading it does
/// not advance the service or change request ownership. No successful token
/// transition creates an error or formats diagnostic text.
pub fn LunaOnlineOpenAIConnectionPool::failure(
  self : LunaOnlineOpenAIConnectionPool,
) -> (String, Error)? {
  self.failure
}
