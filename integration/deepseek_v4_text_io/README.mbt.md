# DeepSeek text boundaries

This checkpoint adapter maps the model's hidden/vocabulary/stream geometry and
epsilon values into `LearnedTextIoPrecision`. It binds `norm.weight` and
`head.weight` as BF16 and `hc_head_fn`, `hc_head_base`, `hc_head_scale` as F32.
No unweighted mean or synthetic learned controls substitute for missing data.

`DeepSeekTextIo` checks combined weight/scratch capacity before streaming and
owns uploaded weights plus the prepared output frame. Borrowed request ports
and output residual storage must outlive its caller-owned execution queue.
Close that queue before this owner; partial preparation and repeated close are
supported. The suffix is reduction → normalization → selected vocabulary rows
→ greedy sample, using the existing queue's completion boundary.

RMS normalization and weight multiplication stay in F32 until the single BF16
activation cast; final logits stay F32. These are explicit precision-IR laws,
not changes to the default GLM text arithmetic. The independent GPU probe
contains witnesses that distinguish both old BF16-rounding paths.

`DeepSeekTextIngress` streams the official BF16 `embed.weight` and replicates
its row over the model's mHC streams through the generic ingress frame. It
needs no vocabulary-head workspace or learned output weights. Prefix/suffix
effects can therefore belong to different stages without duplicating global
weights or adding another queue completion.

These are text components, not a complete decoder/tokenizer frontend,
DSpark runner or full-checkpoint numerical result.
