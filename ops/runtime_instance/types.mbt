///|
/// Exact operator-owner lifecycle; Failed retains cleanup authority.
pub(all) enum RuntimeInstancePhase {
  Starting
  HealthyNotReady
  Ready
  OpenAiQualificationReady
  Draining
  Closed
  Failed
} derive(Debug, Eq)

///|
/// Health is intentionally independent from traffic readiness.
pub(all) enum RuntimeInstanceHealth {
  RuntimeStarting
  RuntimeHealthy
  RuntimeUnhealthy
  RuntimeClosed
} derive(Debug, Eq)

///|
/// Traffic readiness is a separate scalar projection.
pub(all) enum RuntimeInstanceReadiness {
  RuntimeNotReady
  RuntimeReady
} derive(Debug, Eq)

///|
/// Exact, payload-free startup projection retained by the runtime owner. It
/// carries no model identity, request data, filesystem path, or credential.
#valtype
pub struct RuntimeStartupMeasurement {
  priv cold_start_latency_millis : Int
  priv readiness : RuntimeInstanceReadiness
}

///|
/// Qualification readiness never grants production routing authority.
pub(all) enum RuntimeOpenAiQualificationReadiness {
  OpenAiQualificationNotReady
  OpenAiQualificationListenerReady
} derive(Debug, Eq)

///|
/// One bounded cooperative owner transition.
pub enum RuntimeInstanceProgress {
  RuntimeStartingProgress
  RuntimeReadyProgress
  RuntimeServingProgress
  RuntimeDrainingProgress
  RuntimeClosedProgress
  RuntimeFailedProgress
} derive(Debug, Eq)

///|
/// Payload-safe startup and ownership failure stages.
pub(all) enum RuntimeInstanceStage {
  DeploymentBundleAdmission
  MaterializationBinding
  LaunchAdmission
  ModelRootAdmission
  KernelRootAdmission
  PolicyRootAdmission
  DescriptorAdmission
  PolicyAdmission
  TokenizerAdmission
  InstanceAdmission
  PolicyClose
  PhysicalDeviceAdmission
  WorkerExecutableAdmission
  ServicePreparation
  OpenAiQualification
  OpenAiProduction
  SpawnedPhysicalHandoff
  ListenerBinding
  InheritedDrainBinding
  InheritedCredentialBinding
  ColdStartTelemetry
  Cleanup
  Lifecycle
  Clock
} derive(Debug, Eq)

///|
pub fn RuntimeInstanceStage::code(self : RuntimeInstanceStage) -> String {
  match self {
    DeploymentBundleAdmission => "deployment_bundle_admission"
    MaterializationBinding => "materialization_binding"
    LaunchAdmission => "launch_admission"
    ModelRootAdmission => "model_root_admission"
    KernelRootAdmission => "kernel_root_admission"
    PolicyRootAdmission => "policy_root_admission"
    DescriptorAdmission => "descriptor_admission"
    PolicyAdmission => "policy_admission"
    TokenizerAdmission => "tokenizer_admission"
    InstanceAdmission => "instance_admission"
    PolicyClose => "policy_close"
    PhysicalDeviceAdmission => "physical_device_admission"
    WorkerExecutableAdmission => "worker_executable_admission"
    ServicePreparation => "service_preparation"
    OpenAiQualification => "openai_qualification"
    OpenAiProduction => "openai_production"
    SpawnedPhysicalHandoff => "spawned_physical_handoff"
    ListenerBinding => "listener_binding"
    InheritedDrainBinding => "inherited_drain_binding"
    InheritedCredentialBinding => "inherited_credential_binding"
    ColdStartTelemetry => "cold_start_telemetry"
    Cleanup => "cleanup"
    Lifecycle => "lifecycle"
    Clock => "clock"
  }
}

///|
/// Root-free protocol identity selected by authenticated startup policy. It
/// carries no credential, endpoint, transport, or routing authority.
pub enum RuntimeServiceProtocol {
  RuntimeNativeFramedV1
  RuntimeOpenAiResponsesV1
} derive(Debug, Eq)

///|
/// Root-free, non-owning publication of the singular production ingress.
/// Construction is private and never carries a listener, descriptor, root,
/// credential, or control-plane address.
pub struct RuntimeBoundEndpoint {
  priv protocol : RuntimeServiceProtocol
  priv origin : String
} derive(Debug, Eq)

///|
pub fn RuntimeBoundEndpoint::protocol(
  self : RuntimeBoundEndpoint,
) -> RuntimeServiceProtocol {
  self.protocol
}

///|
pub fn RuntimeBoundEndpoint::origin(self : RuntimeBoundEndpoint) -> String {
  self.origin
}

///|
pub(all) suberror RuntimeInstanceError {
  RuntimeInstanceFailed(RuntimeInstanceStage)
  RuntimeInstanceFailedAndCleanup(RuntimeInstanceStage, RuntimeInstanceStage)
} derive(Debug, Eq)

///|
/// Explicit immutable startup file bounds; no hidden process-global defaults.
pub struct RuntimeInstanceFileLimits {
  priv launch : @launch_file.LaunchFileLimits
  priv descriptor : @descriptor_file.RuntimeDescriptorFileLimits
  priv policy : @instance_policy_file.InstancePolicyFileLimits
  priv executable : @worker_executable_file.WorkerExecutableFileLimits
}

///|
/// Truthful native run owner. Lower owners remain singular and opaque.
pub struct RuntimeInstanceOwner {
  priv mut phase : RuntimeInstancePhase
  priv preparations : Array[@online_session.LunaOnlineFramedServicePreparation]
  priv cleanup : Array[@online_session.FailedLunaOnlineInstancePreparation]
  priv services : Array[@online_session.LunaOnlineFramedService]
  priv servers : Array[@online_tcp.LunaOnlineTcpServer]
  priv framed_pools : Array[@online_tcp.LunaOnlineFramedConnectionPool]
  priv openai_servers : Array[@online_tcp.LunaOnlineOpenAIServer]
  priv openai_pools : Array[@online_tcp.LunaOnlineOpenAIConnectionPool]
  priv controls : Array[@online_tcp.LunaOnlineHttpControlServer]
  priv inherited_drains : Array[@inherited_drain.InheritedDrainChannel]
  priv mut inherited_drain_required : Bool
  priv opaque_cli_requires_drain : Bool
  // Root-free release identity derived from the exact admitted inputs and
  // executable consumed by this owner. It is evidence, never filesystem or
  // promotion authority.
  priv release_admission : RuntimeReleaseAdmission
  priv startup_binding : RuntimeOpaqueStartupBinding
  priv borrowed_roots : Array[@approved_fs.ApprovedRoot]
  priv addr : @socket.Addr
  priv transport : @online_tcp.LunaOnlineTcpLimits
  priv telemetry_capacity : Int
  priv connection_capacity : Int
  priv request_work_units : Int
  priv event_work_units : Int
  priv receipt_deadline : @inference.DeadlineBudget
  priv loopback_openai_allowed : Bool
  priv openai_listener_allowed : Bool
  priv clock : @monotonic_clock.MonotonicClock
  priv started_at_millis : UInt64
  priv mut cold_start_recorded : Bool
  priv mut cold_start_latency_millis : Int
}

///|
/// Exact listener selected by authenticated policy plus startup capability.
/// It is private so neither a secret-bearing policy nor a second lifecycle
/// authority can escape the singular runtime owner.
priv enum RuntimeOpaqueStartupBinding {
  RuntimeOpaqueNativeBinding
  RuntimeOpaqueOpenAiBinding(RuntimeOpenAiServerPolicy)
}
