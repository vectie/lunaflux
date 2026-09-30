///|
test "decode module resets object and function macro identities" {
  assert_eq(
    qwen_decode_module_macro_reset(
      "#define LF_HEAD_DIMENSION 128\n#define LF_SCORE_COMPONENT(item) (item * 16)\n#define unrelated 1\n",
    ),
    "#undef LF_HEAD_DIMENSION\n#undef LF_SCORE_COMPONENT",
  )
}
