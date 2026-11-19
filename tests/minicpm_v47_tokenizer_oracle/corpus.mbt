///|
fn corpus(spec : @model.Spec) -> Array[String] raise {
  let cases = [
    "", "Hello, robot!", "12345678901234567890", "-1.25e+10 0xff 2026-11-17", "向前走两步。停止！",
    "こんにちは、ロボット。", "안녕하세요 한글", "مرحبا بالعالم",
    "שלום עולם", "नमस्ते दुनिया", "ภาษาไทย",
    "ქართული", "é e\u{0301} Å A\u{030a} Å", "a\u{0315}\u{0300} a\u{0300}\u{0315}",
    "\u{0301}\u{0323}a", "\u{1100}\u{1161}\u{11a8}", "🙂👩🏽‍🔬👨‍👩‍👧‍👦🇨🇳",
    "can't CAN'T I'm we're I'LL I'd they've", " \t\r\n\n  move\t\tstop\r\n", "a\u{00a0}b\u{2009}c\u{2028}d",
    "a\u{0000}b", "中文English123日本語٤٥٦７８９", "-> [1,2] {\"x\":true}",
    "<image><|image_pad|></image>", "<slice><|image_pad|></slice>\n", "<|im_start|>user\n你好<|im_end|>\n",
    "<think>\n\n</think>\n\n", "<image_id>0</image_id><image><|image_pad|></image_id>",
    "<|unknown_literal|> <image <|im_start <|image_pad|>", "{\"skill_id\":\"stand\",\"parameters\":{},\"reason\":\"前方有障碍\"}",
    "!?!...___--- ++== //\\ @@## $$%% ^^&& **(())",
  ]
  let words = [
    "move", "机器人", "é", "e\u{0301}", "🙂", "12345", "'s", "STOP", "\u{0301}",
  ]
  let separators = ["", " ", "\n", "\r\n", "\t", "\u{00a0}", "<|im_end|>"]
  for left in words {
    for right in words {
      for separator in separators {
        cases.push(left + separator + right)
      }
    }
  }
  let prompt_cases = cases[:].to_owned()
  for text in prompt_cases {
    cases.push(@adapter.render_text_prompt("Select a catalog skill.", text))
    cases.push(@adapter.render_text_prompt("", text, system_present=false))
  }
  for (height, width) in [(224, 224), (448, 896), (896, 896)] {
    let image = @processor.Plan::build(spec, height~, width~).placeholder_text(
      image_id=0,
    )
    cases.push(
      @adapter.render_text_prompt(
        "You control E1 skills.",
        "Look here: " + image + "\nSelect a safe skill.",
      ),
    )
  }
  let seen : Map[String, Bool] = Map([])
  cases.filter(text => {
    if seen.get(text) is Some(_) {
      false
    } else {
      seen[text] = true
      true
    }
  })
}
