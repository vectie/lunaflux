# MiniCPM-V 4.7 text weight startup

This package binds the selected decoder's ordinary BF16 tensors and expert slices
to final device allocations. The existing safetensors inspection/checkpoint index
owns file metadata. The transfer session reuses open shard handles and one
caller-owned scratch buffer; no full host weight arena or payload checksum pass
is introduced. Vision weights are left for the separate visual startup owner.

`Plan` owns the immutable bindings, allocation extents and layer operand indices.
Each routed expert bank replaces the source packed gate/up/down representation;
there is no persistent second routed bank. Ordinary tensors copy their exact
BF16 byte order, including rank-one norms and the rank-three convolution layout.
Embedding and language-head allocations remain distinct.

For the pinned actual headers, the text path binds 693 source tensors into 653
allocations totaling 69,319,779,584 bytes. The 459 vision tensors are outside this
text owner. The 40 routed banks occupy 64,424,509,440 bytes of that total.

`Loaded` owns device allocations and exposes borrowed typed layer/text weights
after startup completes. On partial failure close it before releasing the context.
Close executors and forward frames before `Loaded`, then release the context.
File effects and transfers occur only during prepare, outside token execution.
Real checkpoint contents and GPU numerical results still require target testing.

The header test constructs sparse files with the repository's actual 16 shard
headers and declared extents, then runs the common file inspector and this plan
binder. It never reads/materializes the roughly 70 GB payload. A separate small
transport fixture checks raw BF16 words, reordered matrix slices, untouched
padding, three-byte scratch and session cleanup after a rejected sink. These
tests do not execute `Loaded::prepare` against a real CUDA device.
