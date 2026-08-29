///|
fn external_protocol_path_valid(path : String) -> Bool {
  path.length() > 1 &&
  path.length() <= 256 &&
  path.has_prefix("/") &&
  !path.has_prefix("//") &&
  !path.contains_char(' ') &&
  !path.contains_char('\t') &&
  !path.contains_char('\n') &&
  !path.contains_char('\r') &&
  !path.contains_char('?') &&
  !path.contains_char('#')
}

///|
/// Constructs inert endpoint intent. This never resolves credentials, opens a
/// socket, terminates TLS, or confers health/readiness/routing authority.
pub fn ExternalProtocolPolicyClaims::new(
  mode : ExternalProtocolMode,
  transport_security : ExternalTransportSecurity,
  control_authentication : ExternalControlAuthentication,
  invocation_path : String,
  health_path : String,
  readiness_path : String,
  drain_path : String,
  drain_method : String,
) -> ExternalProtocolPolicyClaims raise ExternalProtocolPolicyError {
  if mode == NativeFramedV1 {
    if transport_security != LoopbackPlaintext ||
      invocation_path != "" ||
      health_path != "" ||
      readiness_path != "" ||
      drain_path != "" ||
      drain_method != "" {
      raise InvalidExternalProtocolPolicy
    }
  } else if invocation_path != "/v1/responses" ||
    health_path != "/healthz" ||
    readiness_path != "/readyz" ||
    !external_protocol_path_valid(drain_path) ||
    drain_path == invocation_path ||
    drain_path == health_path ||
    drain_path == readiness_path ||
    drain_method != "POST" {
    raise InvalidExternalProtocolPolicy
  }
  {
    mode,
    transport_security,
    control_authentication,
    invocation_path,
    health_path,
    readiness_path,
    drain_path,
    drain_method,
  }
}
