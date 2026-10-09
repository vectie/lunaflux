# Prepared shared-KV window frame

Owns bounded ring, metadata, append descriptor and attention output. Four
borrowed launches enter the containing decoder's queue and completion;
there is no additional submit, synchronization or host step allocation.
Close the borrowing queue before this owner and the external operands.
The containing decoder must fold `append_descriptor()[2]` into its existing
error boundary. Request reuse sets a device reset flag; context overflow or
non-contiguous positions poison history until an explicit reset.
