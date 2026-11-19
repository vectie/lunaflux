///|
/// Developer-only cross-repository conformance fixture, not a serving entry.
async fn main {
  let (contract, mode) = match @env.args()[1:] {
    [contract] => (contract, "fp32")
    [contract, "bf16"] => (contract, "bf16")
    [contract, "transformer"] => (contract, "transformer")
    _ => fail("lunachip_fixture CONTRACT_JSON [bf16|transformer]")
  }
  guard @fs.file_size(contract) <= 4096L else { fail("oversized contract") }
  let text = @fs.read_file(contract).text()
  println(
    if mode == "transformer" {
      @fixture.export_transformer_fixture(text)
    } else if mode == "bf16" {
      @fixture.export_bf16_fixture(text)
    } else {
      @fixture.export_fixture(text)
    },
  )
}
