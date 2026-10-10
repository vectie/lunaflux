# Ordered stream-mean capture

The frame owns a bounded segmented output buffer and AOT functions. Counts,
input residuals, module and context are borrowed. `capture` binds a startup
launch for a selected segment; the parent queue appends that launch immediately
after its producer and before recycling the residual. No additional completion,
host round trip, allocation or descriptor upload is required at token execution.

Close the borrowing queue before closing this frame. Preparation checks counts
and the full output budget before allocation; binding rejects input/output alias
and invalid segment ordinals. Native device-double tests cover 32 repeated parent
queue submissions with zero warmed allocations and no blocking synchronization,
partial-submission abort and deterministic resource release.

The generic frame does not know which model layers produce its segments. The
DeepSeek adapter supplies that ordering from its validated DSpark model spec.
