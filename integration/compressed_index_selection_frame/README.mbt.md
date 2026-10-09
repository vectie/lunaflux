# Prepared compressed index selection

Two effects consume transformed query heads, already-scaled head weights and
the published learned compressed cache. All operands and cache metadata are
borrowed. Three private allocations hold reusable score scratch, compressed
row IDs and per-query selected counts, under one workspace ceiling.

Scores are private destructive selection scratch: removing the winning entry
avoids rechecking all previous winners for every candidate. The first effect
rewrites every scratch cell on replay, including inactive rows. The selection
effect resets all output slots to -1 and counts to zero before checking causal
visibility. Returned IDs refer to compressed rows plus the supplied offset;
they are never expanded to raw token IDs.

No queue, host synchronization, filesystem access or token-step allocation is
introduced. Close the containing executor first, then this frame, then its
borrowed operands. GPU numerical validation is separate from fake-driver
ownership and allocation tests.
