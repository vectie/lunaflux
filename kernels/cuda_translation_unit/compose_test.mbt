///|
test "CUDA fragments preserve directives comments ordering and nested composition" {
  let first = b"void a() {}"
  let second = b"#include <stdint.h>\n// final comment"
  let third = b"void b() {}\n"
  let combined = @cuda_translation_unit.compose([first, b"", second, third])
  assert_eq(
    combined, b"void a() {}\n#include <stdint.h>\n// final comment\nvoid b() {}\n",
  )
  assert_eq(
    combined,
    @cuda_translation_unit.compose([
      @cuda_translation_unit.compose([first, second]),
      third,
    ]),
  )
  assert_eq(
    combined,
    @cuda_translation_unit.compose([
      first,
      @cuda_translation_unit.compose([second, third]),
    ]),
  )
  assert_eq(@cuda_translation_unit.compose([combined]), combined)
  assert_eq(@cuda_translation_unit.compose([]), b"")
  assert_eq(@cuda_translation_unit.compose([b"", b""]), b"")
}
