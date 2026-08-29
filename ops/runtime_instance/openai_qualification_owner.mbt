///|
/// Consumes the one already-prepared service into the existing OpenAI
/// Responses server exactly once. This qualification-only composition accepts
/// an opaque caller-admitted authentication capability and is restricted to
/// an explicitly configured loopback listener.
async fn RuntimeInstanceOwner::bind_openai_qualification(
  self : RuntimeInstanceOwner,
  policy : RuntimeOpenAiQualificationPolicy,
) -> Unit raise RuntimeInstanceError {
  self.bind_openai_owner(policy.server, OpenAiQualificationBinding)
}

///|
/// Separate loopback qualification readiness; never production readiness.
pub fn RuntimeInstanceOwner::openai_qualification_readiness(
  self : RuntimeInstanceOwner,
) -> RuntimeOpenAiQualificationReadiness {
  if self.phase == OpenAiQualificationReady &&
    self.openai_servers.length() == 1 &&
    self.servers.length() == 0 &&
    self.controls.length() == 0 &&
    self.openai_servers[0].state() != LunaOnlineTcpServerDraining &&
    self.openai_servers[0].state() != LunaOnlineTcpServerClosed {
    OpenAiQualificationListenerReady
  } else {
    OpenAiQualificationNotReady
  }
}
