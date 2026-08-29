///|
/// Consumes the singular prepared service into the existing OpenAI Responses
/// server and publishes ordinary production readiness for embedding callers.
///
/// This path requires a distinct production policy containing an opaque
/// caller-built Bearer policy and explicit loopback-ingress approval. It does
/// not choose a CLI secret transport, terminate TLS, or implement HTTP control
/// routes. Those remain the embedding deployment owner's responsibility.
pub async fn RuntimeInstanceOwner::bind_openai_production(
  self : RuntimeInstanceOwner,
  policy : RuntimeOpenAiProductionPolicy,
) -> Unit raise RuntimeInstanceError {
  self.bind_openai_owner(
    policy.server,
    OpenAiEmbeddingProductionBinding(policy.ingress),
  )
}

///|
/// Returns only the loopback address owned by an embedding-production OpenAI
/// server that has published ordinary readiness.
pub fn RuntimeInstanceOwner::openai_production_local_addr(
  self : RuntimeInstanceOwner,
) -> @socket.Addr raise RuntimeInstanceError {
  if self.readiness() != RuntimeReady ||
    self.openai_servers.length() != 1 ||
    self.servers.length() != 0 {
    raise fail(OpenAiProduction)
  }
  self.openai_servers[0].local_addr()
}
