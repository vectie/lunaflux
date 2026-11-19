# MiniCPM-V 4.7 BF16 weight binding

Describes and binds all 1,152 tensors of the selected ordinary decoder and vision
checkpoint. Packed expert axes, doubled gated-query projection, recurrent
parameters, both vision mergers and every vision-layer bias are explicit.
Binding consumes common validated safetensors metadata, not weight payloads.
The result supplies stable tensor ordinals and shard/file ranges for preparation.

Fixtures are the exact padded safetensors headers from all 16 shards of
`openbmb/MiniCPM-V-4.7-35B-A3B` revision
`98e79ba972b586ac6146d3505771ce1467e199ba`. They contain names, types, shapes and
ranges, **no tensor values**. The upstream model is Apache-2.0 licensed. Retain
header padding: its byte count participates in absolute tensor file offsets.

The real metadata totals 70,425,751,648 bytes. Header/shape tests do not verify
weight contents or numerical model outputs. No MTP tensors occur in this source;
the ordinary route does not claim the separate DSpark draft model.
