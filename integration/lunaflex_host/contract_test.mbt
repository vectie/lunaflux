///|
test "optional host links LunaFlux and accepts sibling LunaFlex" {
  inspect(@lunaflux_host.linked_core_version(), content="0.1.0")
  assert_eq(@lunaflux_host.linked_lunaflux_contract_probe(), 1UL)
  assert_eq(@lunaflux_host.required_core_abi_version(), 1)
  assert_true(@lunaflux_host.is_linked_core_compatible())
  @lunaflux_host.require_linked_core()
  let smoke = @smoke.run_reference_smoke()
  assert_eq(smoke.operation_count(), 12)
  assert_eq(smoke.generated_tokens(), [0, 0, 0])
}
