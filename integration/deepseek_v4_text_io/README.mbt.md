# DeepSeek learned text output

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

This is an output component, not a complete decoder, embedding frontend,
DSpark runner or full-checkpoint numerical result.
