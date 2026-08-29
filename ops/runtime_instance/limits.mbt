///|
pub fn RuntimeInstanceFileLimits::new(
  launch~ : @launch_file.LaunchFileLimits,
  descriptor~ : @descriptor_file.RuntimeDescriptorFileLimits,
  policy~ : @instance_policy_file.InstancePolicyFileLimits,
  executable~ : @worker_executable_file.WorkerExecutableFileLimits,
) -> RuntimeInstanceFileLimits {
  { launch, descriptor, policy, executable }
}
