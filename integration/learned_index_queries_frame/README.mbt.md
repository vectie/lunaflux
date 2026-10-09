# Prepared learned index queries

Three projection/scaling effects followed by three existing transform effects.
Counts, positions, hidden, normalized low-rank query input and two checkpoint
banks are borrowed. The owner accounts for its three BF16 intermediates and
the composed transform's two buffers under one scratch ceiling.

No queue or submission boundary is added. Preparation validates all borrowed
spans before allocating scratch. Close the containing queue, this owner, then
external weights/inputs. Warm launches are immutable borrowed descriptors.
