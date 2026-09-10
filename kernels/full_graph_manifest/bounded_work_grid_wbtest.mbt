///|
test "optional compiler grid preserves old manifest bytes and round trips" {
  let identity = full_manifest_entry(0)
  let dimensions = full_manifest_dimensions(191)
  let render = fn(grid : Int?) -> String {
    let output = StringBuilder()
    let count = append_manifest_modules(output, [
      {
        digest: identity.family().module_digest(),
        path: "module.cubin",
        entries: [
          {
            family_id: identity.family().family_id(),
            entry_point_id: identity.entry_point_id(),
            function_symbol: "matrix",
            companion_dimensions: None,
            row_variants: [
              @artifact.KernelRowVariant::new(
                maximum_token_rows=16,
                function_symbol="rows16",
                dimensions~,
                bounded_work_grid_x=grid,
              ),
            ],
          },
        ],
      },
    ])
    ignore(count)
    output.to_string()
  }
  let digest = identity.family().module_digest().sha256()
  let legacy = "\"modules\":[{\"sha256\":\"\{digest}\",\"path\":\"module.cubin\",\"entry_points\":[{\"family_id\":11,\"entry_point_id\":1,\"symbol\":\"matrix\",\"row_variants\":[{\"maximum_token_rows\":16,\"symbol\":\"rows16\",\"grid_x\":192,\"block_x\":32,\"shared_memory_bytes\":0}]}]}],"
  assert_eq(render(None), legacy)
  let bounded = render(Some(48))
  assert_eq(
    bounded,
    legacy.replace(
      old="\"shared_memory_bytes\":0}",
      new="\"shared_memory_bytes\":0,\"bounded_work_grid_x\":48}",
    ),
  )
  let document = "{\{bounded}\"end\":0}"
  guard @json.parse(document) is Object(root) else { fail("manifest root") }
  guard root["modules"] is Array(modules) else { fail("modules") }
  guard modules[0] is Object(module_fields) else { fail("module") }
  guard module_fields["entry_points"] is Array(entries) else { fail("entries") }
  guard entries[0] is Object(fields) else { fail("entry") }
  assert_eq(
    @artifact_file.decode_row_variants(fields["row_variants"], 64)[0].bounded_work_grid_x(),
    Some(48),
  )
}
