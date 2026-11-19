# Full-model numerical reference

`oracle.cpp` is an offline CPython transport adapter. It loads the full selected
checkpoint with unchanged Transformers5.18.0 and PyTorch, using eager attention,
BF16 checkpoint dtype, the actual CUDA device and no gradients. Python/Torch
remain outside the production LunaFlux runtime.

Inputs are the native CLI's prepared text-plan token IDs. Each request resets
the upstream cache/rope state, uses32-token cached prefill boundaries, then
greedy decoding with the pinned two EOS IDs. The adapter saves raw FP32-converted
upstream logits and sampled/emitted IDs. `requests.json` covers a short prompt,
a58-token prompt spanning two prefill chunks, decoder steps and request reuse.
This input scope does not independently qualify chat rendering; that has a
separate pinned template/tokenizer reference.

Build with the reference environment's Python headers and embed library:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I/usr/include/python3.12 oracle.cpp -lpython3.12 -ldl -lm -o minicpm-model-oracle
./minicpm-model-oracle ACTUAL_MODEL_ROOT NATIVE_PLAN.json NEW_REFERENCE_OUTPUT
moon run scripts/compare-minicpm-v47-model.mbtx NATIVE_OUTPUT NEW_REFERENCE_OUTPUT/report.json NEW_COMPARISON.json
```

Set the reference's isolated Transformers dependency path and offline HF mode
when launching it. On shared-memory Spark also set
`HF_DEACTIVATE_ASYNC_LOAD=1` and `PYTORCH_NO_CUDA_MEMORY_CACHING=1`. The adapter
loads CPU-backed BF16 weights with upstream `from_pretrained`, then moves the
same model to CUDA and checks every parameter's device before inference.
This avoids retaining CPU source and GPU destination copies for the whole
checkpoint during direct-device loading. Sequential materialization and
allocator settings change loading/resource behavior, not forward arithmetic.
Keep the offline container memory bounded. The initial asynchronous reference
loader was OOM-killed before inference; a subsequent direct-device attempt was
stopped before pressure became critical. Neither partial log is comparison
evidence. Run the native resident and upstream model sequentially;
each full checkpoint occupies about70GB before mutable/driver storage.
The comparison reports exact greedy-token agreement and finish reasons. When
the native diagnostic's `logits.json` is present, it also reads complete
248,144-word logit rows, rejects non-finite values, reports maximum/mean/RMS
absolute differences and compares argmax including EOS. After a token mismatch,
only rows sharing the same preceding token prefix are compared. These measured
differences are not an equivalence proof or quality acceptance. Image execution
and planner quality remain separate gates. Compiling this adapter alone
establishes no model result.

The native executable in this package composes the existing public weight,
module, storage, frame and request owners. It saves the full selected FP32
logit row after each completed sampling step, including EOS. Diagnostic file
writes and device readbacks stay outside production owners; its timing is
unsuitable for performance claims. All resources close explicitly in dependency
order. It requires the actual BF16 sm121 device and full checkpoint:

```sh
moon run tests/minicpm_v47_model_oracle --release --target native -- MODEL_ROOT CUBIN_ROOT tests/minicpm_v47_model_oracle/requests.extended.json NEW_NATIVE_OUTPUT
```

The extended queue covers repeated short requests, a prefill spanning multiple
32-row chunks, multi-step counting and a structured robot-skill proposal. It
does not execute a skill or establish planner quality from one proposal.

`scripts/stage-minicpm-v47-vision-fixture.mbtx NEW_OUTPUT` creates lossless
448x448 red/blue-square PNGs and a media queue, using macOS `sips` only during
fixture construction. The queue asks the same color question for each image,
then a text-only request to check image-state clearing. Filenames and expected
colors are not included in model prompts. This is synthetic grounding evidence,
not a camera or physical robot test.
