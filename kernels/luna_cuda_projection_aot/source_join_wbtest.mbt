///|
test "CUDA fragment joins delimit braces comments and preprocessor directives" {
  for ending in ["}", "}\n", "} // final fragment"] {
    let joined = @utf8.decode(
      projection_join_source_fragments(
        @utf8.encode(ending),
        "#include <cuda_bf16.h>\nvoid next() {}",
      ),
    )
    assert_true(joined.contains("\n#include <cuda_bf16.h>\n"))
    assert_true(joined.has_suffix("\n"))
    assert_false(joined.contains("}#include"))
  }
  let primary = @utf8.encode("void unchanged() {}")
  assert_eq(projection_join_source_fragments(primary, ""), primary)
  let builder = StringBuilder()
  projection_append_source_fragment(builder, "void first() {}")
  projection_append_source_fragment(
    builder, "#include <stdint.h>\nvoid second() {}",
  )
  assert_true(builder.to_string().contains("\n#include <stdint.h>"))
}
