# Checkpoint Jinja rendering comparison

This optional offline test executes the selected checkpoint's actual
`chat_template.jinja` with Jinja2, through the unmodified
[jinja2-cli](https://github.com/mattrobenolt/jinja2-cli) command. MoonBit constructs
inputs and compares exact rendered strings from both the native prompt helper
and model-owned HTTP normalization. No agent-authored Python runs, and no Jinja
or Python dependency enters the production runtime.

```sh
moon run scripts/verify-minicpm-v47-chat-template.mbtx ABSOLUTE_REPO PINNED_ASSETS JINJA2_CLI NEW_OUTPUT
```

The local offline environment uses `jinja2-cli==1.0.1`, `Jinja2==3.1.6` and
CPython 3.12. The wrapper prints actual CLI/Jinja versions into the run log.
Transformers uses `trim_blocks=true`, `lstrip_blocks=true` and
`keep_trailing_newline=false`. The CLI always keeps a trailing source newline,
so the test drops only the template's final LF in its test copy to reproduce
that environment setting. The checkpoint source is otherwise unchanged.

The 128 cases cover absent, empty and nonempty system messages; empty and
multilingual strings; NFC input, NUL, literal image markers; text-content lists;
all 29 Python whitespace characters and four visually similar characters that
must be preserved. Both native rendering and normalized HTTP messages must match.
Per-case variables and upstream/native outputs are retained for review.

Scope is optional system + one user, no tools/history, thinking disabled and
generation cue enabled. This is not full Transformers model execution or proof
of image expansion/slicing, tokenizer, logits, GPU execution or model quality.
