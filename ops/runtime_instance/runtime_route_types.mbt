///|
/// Closed internal descriptor route selected only by the authenticated launch
/// recipe. Neither branch can be cross-decoded or retried through the other.
priv enum AdmittedRuntimeDescriptor {
  LegacyRuntimeDescriptor(@descriptor_file.RuntimeDescriptorAdmission)
  I8RuntimeDescriptor(@descriptor_file.I8RuntimeDescriptorAdmission)
  Fp8RuntimeDescriptor(@descriptor_file.Fp8RuntimeDescriptorAdmission)
  TensorParallelRuntimeDescriptor(
    @descriptor_file.TensorParallelRuntimeDescriptorAdmission
  )
}

///|
/// Closed pure instance join paired with its exact descriptor family.
priv enum AdmittedRuntimeInstance {
  LegacyRuntimeInstance(@instance_admission.RuntimeInstanceAdmission)
  I8RuntimeInstance(@instance_admission.I8RuntimeInstanceAdmission)
  Fp8RuntimeInstance(@instance_admission.Fp8RuntimeInstanceAdmission)
  TensorParallelRuntimeInstance(
    @instance_admission.TensorParallelRuntimeInstanceAdmission
  )
}

///|
/// Common listener fields projected from the same admitted instance that is
/// consumed by service preparation. This is inert and owns no socket.
priv struct RuntimeServiceEnvelope {
  addr : @socket.Addr
  transport : @online_tcp.LunaOnlineTcpLimits
  telemetry_capacity : Int
  connection_capacity : Int
  request_work_units : Int
  event_work_units : Int
  receipt_deadline : @inference.DeadlineBudget
  loopback_openai_allowed : Bool
  openai_listener_allowed : Bool
}
