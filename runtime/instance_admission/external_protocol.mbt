///|
fn loopback_listener(host : StringView) -> Bool {
  host == "127.0.0.1" || host == "localhost" || host == "::1"
}

///|
fn private_network_listener(host : StringView) -> Bool {
  host == "0.0.0.0" || host == "::"
}

///|
/// Joins authenticated endpoint intent to the service policy available to the
/// singular runtime owner. Deployment TLS remains external authority; a
/// configured OpenAI policy represents only the local loopback composition.
fn external_protocol_admitted(
  protocol : @instance_policy_file.ExternalProtocolPolicyClaims,
  service : @service.ServiceConfig,
  openai_service_available? : Bool = false,
) -> Bool {
  let transport_admitted = match protocol.transport_security() {
    LoopbackPlaintext => loopback_listener(service.listen_host())
    PrivateNetworkPlaintext =>
      protocol.mode() == OpenAiResponsesV1 &&
      private_network_listener(service.listen_host())
    DeploymentTls => false
  }
  transport_admitted &&
  (match protocol.mode() {
    NativeFramedV1 => true
    OpenAiResponsesV1 => openai_service_available
  })
}
