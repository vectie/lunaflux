///|
fn external_protocol_service(host : String) -> @service.ServiceConfig {
  try! @service.new(host, 8443, 4096, 1000, Normal)
}

///|
fn openai_protocol(
  security : @instance_policy_file.ExternalTransportSecurity,
) -> @instance_policy_file.ExternalProtocolPolicyClaims {
  try! @instance_policy_file.ExternalProtocolPolicyClaims::new(
    OpenAiResponsesV1,
    security,
    DeploymentBearer,
    "/v1/responses",
    "/healthz",
    "/readyz",
    "/v1/drain",
    "POST",
  )
}

///|
test "native protocol remains admitted without inventing HTTP authority" {
  let protocol = try! @instance_policy_file.ExternalProtocolPolicyClaims::new(
    NativeFramedV1,
    LoopbackPlaintext,
    DeploymentBearer,
    "",
    "",
    "",
    "",
    "",
  )
  let admitted = external_protocol_admitted(
    protocol,
    external_protocol_service("127.0.0.1"),
    openai_service_available=false,
  )
  assert_true(admitted)
  let exposed = external_protocol_admitted(
    protocol,
    external_protocol_service("0.0.0.0"),
  )
  assert_false(exposed)
}

///|
test "OpenAI loopback intent requires the local service policy" {
  let admitted = external_protocol_admitted(
    openai_protocol(LoopbackPlaintext),
    external_protocol_service("127.0.0.1"),
  )
  assert_false(admitted)
}

///|
test "OpenAI loopback intent is admitted only with exact service claims" {
  let admitted = external_protocol_admitted(
    openai_protocol(LoopbackPlaintext),
    external_protocol_service("127.0.0.1"),
    openai_service_available=true,
  )
  assert_true(admitted)
}

///|
test "OpenAI exposure fails closed for non-loopback and unapproved TLS" {
  let non_loopback = external_protocol_admitted(
    openai_protocol(LoopbackPlaintext),
    external_protocol_service("0.0.0.0"),
  )
  assert_false(non_loopback)
  let tls = external_protocol_admitted(
    openai_protocol(DeploymentTls),
    external_protocol_service("0.0.0.0"),
  )
  assert_false(tls)
}

///|
test "private-network plaintext admits only OpenAI wildcard listeners" {
  assert_true(
    external_protocol_admitted(
      openai_protocol(PrivateNetworkPlaintext),
      external_protocol_service("0.0.0.0"),
      openai_service_available=true,
    ),
  )
  assert_true(
    external_protocol_admitted(
      openai_protocol(PrivateNetworkPlaintext),
      external_protocol_service("::"),
      openai_service_available=true,
    ),
  )
  for host in ["127.0.0.1", "localhost", "192.168.2.8"] {
    assert_false(
      external_protocol_admitted(
        openai_protocol(PrivateNetworkPlaintext),
        external_protocol_service(host),
        openai_service_available=true,
      ),
    )
  }
}

///|
test "native protocol cannot construct private-network authority" {
  let accepted = try
    @instance_policy_file.ExternalProtocolPolicyClaims::new(
      NativeFramedV1,
      PrivateNetworkPlaintext,
      DeploymentBearer,
      "",
      "",
      "",
      "",
      "",
    )
  catch {
    _ => false
  } noraise {
    _ => true
  }
  assert_false(accepted)
}
