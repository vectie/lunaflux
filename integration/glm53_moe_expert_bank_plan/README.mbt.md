# GLM-5.3 MoE expert-bank candidate plan

This integration boundary joins the exact GLM checkpoint manifest and its
per-expert tensor names to the routed- and shared-expert correctness candidate
ABIs. It authenticates the model identity, manifest layout, layer, six bank
layouts, operand roles and sizes, and candidate recipes. Every returned plan is
non-materializing, non-uploading, and non-launching evidence.
