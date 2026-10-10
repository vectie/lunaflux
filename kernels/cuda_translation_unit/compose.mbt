///|
/// Compose generated CUDA fragments in order, preserving their bytes and
/// terminating each nonempty fragment's final line. Preprocessor directives and
/// trailing line comments therefore cannot consume the next fragment. This is
/// an offline source transformation, not compilation or runtime submission.
///
/// ```mbt check
/// test {
///   assert_eq(
///     compose([b"void a() {}", b"#include <stdint.h>"]),
///     b"void a() {}\n#include <stdint.h>\n",
///   )
/// }
/// ```
pub fn compose(fragments : ArrayView[Bytes]) -> Bytes {
  let output = @buffer.Buffer()
  for fragment in fragments {
    if !fragment.is_empty() {
      output.write_bytes(fragment)
      if fragment[fragment.length() - 1] != b'\n' {
        output.write_byte(b'\n')
      }
    }
  }
  output.to_bytes()
}
