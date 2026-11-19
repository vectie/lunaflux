///|
priv struct Case {
  system : String
  user : String
  system_present : Bool
  parts : Array[String]?
}

///|
fn corpus() -> Array[Case] {
  let cases = []
  for
    user in [
      "", "move", "向前走两步。", " \nmove\r\n ", "e\u{0301}🙂", "a\u{0000}b",
      "<image><|image_pad|></image>",
    ] {
    for
      (system, system_present) in [
        ("Select a skill.", true),
        ("", true),
        ("", false),
      ] {
      cases.push({ system, user, system_present, parts: None, })
    }
  }
  for
    space in [
      "\u{0009}", "\u{000a}", "\u{000b}", "\u{000c}", "\u{000d}", "\u{001c}", "\u{001d}",
      "\u{001e}", "\u{001f}", " ", "\u{0085}", "\u{00a0}", "\u{1680}", "\u{2000}",
      "\u{2001}", "\u{2002}", "\u{2003}", "\u{2004}", "\u{2005}", "\u{2006}", "\u{2007}",
      "\u{2008}", "\u{2009}", "\u{200a}", "\u{2028}", "\u{2029}", "\u{202f}", "\u{205f}",
      "\u{3000}",
      // These look similar but Python/Jinja do not strip them.
       "\u{180e}", "\u{200b}", "\u{2060}", "\u{feff}",
    ] {
    for
      (system, system_present) in [
        (space + "policy" + space, true),
        (space, true),
        ("", false),
      ] {
      cases.push({
        system,
        user: space + "move" + space,
        system_present,
        parts: None,
      })
    }
  }
  for
    parts in [
      ["  one", "two  "],
      ["", "", ""],
      ["\u{00a0}机器人", "🙂\u{3000}"],
      ["a\r\nb", "c\n\nd"],
    ] {
    cases.push({
      system: " policy ",
      user: parts.join("\n"),
      system_present: true,
      parts: Some(parts),
    })
    cases.push({
      system: "",
      user: parts.join("\n"),
      system_present: false,
      parts: Some(parts),
    })
  }
  cases
}
