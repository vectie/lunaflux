///|
/// A contiguous noncausal attention document. Different documents must never
/// share the softmax reduction. Offsets are token rows, not tensor elements.
pub struct AttentionDocument {
  start : Int
  count : Int
}

///|
/// Compact attention visibility: live text, references and both target
/// modalities form one document; padding forms a separate document. This
/// matches cumulative boundaries [0, used, total] without a quadratic mask.
pub fn Sequence::attention_documents(
  self : Self,
) -> ReadOnlyArray[AttentionDocument] {
  let total = self.assembly.rows().length()
  let documents : Array[AttentionDocument] = [{ start: 0, count: self.used, }]
  if total > self.used {
    documents.push({ start: self.used, count: total - self.used, })
  }
  ReadOnlyArray::from_array(documents)
}

///|
/// Refiner attention covers only the live text prefix. The rest of the packed
/// buffer is not a second refiner document and must not be submitted here.
pub fn Sequence::refiner_document(self : Self) -> AttentionDocument {
  { start: 0, count: self.segments[0].count, }
}

///|
/// Cumulative boundaries for generic packed-attention lowering. Empty trailing
/// padding is omitted so every submitted document has a nonempty reduction.
pub fn Sequence::attention_boundaries(self : Self) -> ReadOnlyArray[Int] {
  let total = self.assembly.rows().length()
  ReadOnlyArray::from_array(
    if total == self.used {
      [0, self.used]
    } else {
      [0, self.used, total]
    },
  )
}
