# Selected MiniCPM checkpoint fixture

`with_checkpoint` stages the actual 16 BF16 headers with sparse zero payload
holes, opens a scoped root and passes parsed model/checkpoint metadata to a
test callback. The callback borrows the root; the helper closes it and removes
the temporary files afterward.

These are metadata and byte-transfer fixtures. They contain no checkpoint
weights and must never support a model arithmetic or inference claim. Product
packages use this helper only through test imports.
