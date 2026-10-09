# Retained compact rows

CUDA lowering for the backend-neutral `RetainedRowsPrecision` plan. Cells are
copied as exact 16-bit words; this package neither quantizes nor interprets them.
The containing ordered executor owns submission and completion.

Three effects reserve a contiguous append, copy rows, then publish the retained
length. Reservation validates all positions before any cache write. Overflow,
gaps and an upstream error expose no readable rows and retain a sticky error;
an explicit request reset clears history. Empty appends preserve prior rows.
A zero logical capacity uses one inaccessible physical sentinel row.

Native source/plan tests and fake prepared execution cover bounded allocation,
error ports and deterministic release. Physical numerical/sanitizer verification
is still pending; source checks are not GPU correctness evidence.
