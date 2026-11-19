///|
async fn main {
  guard @env.args()[1:] is [assets, jinja, output] else {
    fail(
      "usage: minicpm_v47_chat_template_oracle PINNED_ASSETS JINJA2_CLI NEW_OUTPUT",
    )
  }
  @fs.mkdir(output, recursive=true)
  let model = @model.Spec::parse(
    @fs.read_file(assets + "/config.json").binary()[:],
  )
  let original = @fs.read_file(assets + "/chat_template.jinja").text()
  // The CLI keeps the source's final newline; Transformers' Jinja environment
  // drops one. Removing only that EOF newline reproduces keep_trailing_newline=false.
  let source = if original.has_suffix("\n") {
    original[:original.length() - 1].to_owned()
  } else {
    original
  }
  let template = output + "/chat_template.jinja"
  @fs.write_file(template, source, create_mode=CreateNew)
  let cases = corpus()
  let mut bytes_compared = 0
  for index, case in cases {
    let messages : Array[Json] = []
    if case.system_present {
      messages.push({ "role": "system", "content": Json(case.system) })
    }
    let content : Json = match case.parts {
      None => Json(case.user)
      Some(parts) =>
        Json(
          parts.map(text => {
            let part : Json = { "type": "text", "text": Json(text) }
            part
          }),
        )
    }
    messages.push({ "role": "user", "content": content })
    let variables : Json = {
      "messages": Json(messages),
      "tools": [],
      "enable_thinking": false,
      "add_generation_prompt": true,
    }
    let prefix = output + "/case-\{index}"
    @fs.write_file(
      prefix + ".json",
      variables.stringify(),
      create_mode=CreateNew,
    )
    guard @process.run(jinja, [
        template,
        prefix + ".json",
        "--format=json",
        "--trim-blocks",
        "--lstrip-blocks",
        "--outfile",
        prefix + ".txt",
      ]) ==
      0 else {
      fail("upstream Jinja rendering failed at case \{index}")
    }
    let reference = @fs.read_file(prefix + ".txt").text()
    let native = @adapter.render_text_prompt(
      case.system,
      case.user,
      system_present=case.system_present,
    )
    @fs.write_file(prefix + ".native.txt", native, create_mode=CreateNew)
    guard native == reference else {
      fail(
        "template differs at case \{index}: native=\{Json(native).stringify()}, Jinja=\{Json(reference).stringify()}",
      )
    }
    let body : Json = {
      "model": "MiniCPM-V-4.7-35B-A3B",
      "messages": Json(messages),
    }
    let normalized = @media.normalize_chat(
      model,
      @utf8.encode(body.stringify()),
    )
    guard @json.parse(@utf8.decode(normalized.body()))
      is { "messages": Array(normalized_messages), .. } &&
      normalized_messages.last() is Some({ "content": String(user), .. }) else {
      fail("normalization lost messages")
    }
    let system = if case.system_present {
      guard normalized_messages[0] is { "content": String(system), .. } else {
        fail("normalization lost system")
      }
      system
    } else {
      ""
    }
    guard @adapter.render_text_prompt(
        system,
        user,
        system_present=case.system_present,
      ) ==
      reference else {
      fail("HTTP normalization differs at case \{index}")
    }
    bytes_compared += @utf8.encode(reference).length() * 2
  }
  let report : Json = {
    "passed": true,
    "cases": cases.length(),
    "bytes_compared": bytes_compared,
    "template": "selected checkpoint chat_template.jinja",
    "declared_asset_revision": @fs.read_file(assets + "/revision.txt")
    .text()
    .trim()
    .to_owned(),
    "profile": "optional system + user; string/text list; no tools/history/thinking",
    "source_final_newline_dropped_to_match_transformers_environment": original.has_suffix(
      "\n",
    ),
    "gpu_execution": false,
  }
  @fs.write_file(
    output + "/report.json",
    report.stringify(indent=2),
    create_mode=CreateNew,
  )
  println(report.stringify(indent=2))
}
