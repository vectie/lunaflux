///|
pub extend TensorField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend TensorField with Eq::{not_equal, equal}

///|
pub extend SafetensorsReaderError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend SafetensorsReaderError with Eq::{not_equal, equal}
