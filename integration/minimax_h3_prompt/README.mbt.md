# H3 prompt presentation

This startup-only MoonBit adapter follows local SGLang H3 `presentation.py`.
H3 uses **no chat template**: labels and the verbatim prompt are tokenized as
separate segments without inserted BOS, EOS, system or assistant messages.

- `Text`: the prompt alone.
- `Keyframes`: semantic signatures `[0]`, `[-1]`, `[0,-1]`; ordered Picture
  labels and image blocks followed by the prompt.
- `References`: request-order image, audio and video conditions. Audio adds
  only its label to Qwen; video adds a timestamp and a video block per temporal
  patch. Ordinals are independent for each media kind.

Inputs are admitted tokenizer/config plus already decoded and preprocessed
visual grids and video **block** timestamps. This package does not decode
files, resize pixels, choose reference frames or implement the vision encoder.
The vision adapter must produce the same merged feature rows in the returned
grid order. Returned `VisualPlacement` is directly accepted by the existing
text-encoder request; it checks every image/video placeholder has one visual
row. `token_tags` includes the enclosing vision delimiters as VIDEO (0), with
labels, timestamps and prompt text tagged TEXT (1), matching AdaLN input.

Tokenization reuses the production `TokenizerSpec` and rejects overflow rather
than truncating media slots. Timestamp formatting rounds the exact binary
Double to one decimal with ties-to-even, including decimal-edge cases such as
0.15, following Python `.1f` rather than a rounded floating multiply by ten.

Tests use a deterministic byte-vocabulary fixture with admitted special-token
IDs to check exact presentation sequences, tags and placement. They do not
claim numerical encoder equivalence or a real checkpoint tokenizer benchmark.
