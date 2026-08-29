///|
pub fn ExternalProtocolPolicyClaims::mode(
  self : ExternalProtocolPolicyClaims,
) -> ExternalProtocolMode {
  self.mode
}

///|
pub fn ExternalProtocolPolicyClaims::transport_security(
  self : ExternalProtocolPolicyClaims,
) -> ExternalTransportSecurity {
  self.transport_security
}

///|
pub fn ExternalProtocolPolicyClaims::control_authentication(
  self : ExternalProtocolPolicyClaims,
) -> ExternalControlAuthentication {
  self.control_authentication
}

///|
pub fn ExternalProtocolPolicyClaims::invocation_path(
  self : ExternalProtocolPolicyClaims,
) -> String {
  self.invocation_path
}

///|
pub fn ExternalProtocolPolicyClaims::health_path(
  self : ExternalProtocolPolicyClaims,
) -> String {
  self.health_path
}

///|
pub fn ExternalProtocolPolicyClaims::readiness_path(
  self : ExternalProtocolPolicyClaims,
) -> String {
  self.readiness_path
}

///|
pub fn ExternalProtocolPolicyClaims::drain_path(
  self : ExternalProtocolPolicyClaims,
) -> String {
  self.drain_path
}

///|
pub fn ExternalProtocolPolicyClaims::drain_method(
  self : ExternalProtocolPolicyClaims,
) -> String {
  self.drain_method
}
