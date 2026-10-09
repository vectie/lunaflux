# Official GLM-5.3 CUDA AOT integration evidence

These tests join the official full and Flash specifications, exact BF16 tensor
manifests, logical numeric plans, family-neutral GLM kernel requirements, and
non-bindable CUDA candidates. Production packages remain acyclic and do not
import the family-specific manifest or numeric owners.

The test join proves the current LM-head, DSA attention-output, and Flash KDA
projection/decay-control/recurrent/gated-normalization and Full/Flash F32 MoE
router-projection candidates agree with exact tensor names, shapes, dtypes,
producer/consumer byte geometry, and execution identity.
The routing join additionally authenticates the official one-group/top-two
group score, one selected group, expert top-eight, sum-plus-`1e-20`
normalization, and `2.5` scaling boundary for both profiles.
It does not place the manifest layout digest or numeric binding digest inside a
candidate and grants no artifact, device, launch, or qualification authority.
