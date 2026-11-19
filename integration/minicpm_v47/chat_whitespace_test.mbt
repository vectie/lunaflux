///|
test "pinned Jinja whitespace is trimmed while zero-width content is retained" {
  assert_eq(
    @minicpm_v47.trim_message_content(
      "\u{000b}\u{001c}\u{0085}\u{00a0}move\u{3000}",
    ),
    "move",
  )
  assert_eq(
    @minicpm_v47.trim_message_content("\u{200b}move\u{feff}"),
    "\u{200b}move\u{feff}",
  )
  assert_eq(
    @minicpm_v47.render_text_prompt(
      "\u{00a0}policy\u{3000}", "\u{000b}move\u{0085}",
    ),
    "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nmove<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  )
}
