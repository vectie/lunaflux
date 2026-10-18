///|
/// Allocating teardown-only observation; never called by normal token progress.
pub fn LunaOnlineOpenAIConnectionPool::drain_diagnostic(self : Self) -> String {
  let mut slots = ""
  for index, slot in self.slots {
    if slot.phase != LUNA_OPENAI_POOL_FREE {
      slots += " slot=\{index}:\{slot.phase}:events=\{slot.semantic_events.length()}:rejections=\{slot.rejections.length()}:routes=\{slot.routes.length()}"
    }
  }
  "connections=\{self.active_connections} failure=\{Repr(self.failure)} " +
  self.service().drain_diagnostic() +
  slots
}
