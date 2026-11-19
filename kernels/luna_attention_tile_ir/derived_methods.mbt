///|
pub extend AttentionTileInputRole with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileInputRole with Eq::{not_equal, equal}

///|
pub extend AttentionTileValueKind with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileValueKind with Eq::{not_equal, equal}

///|
pub extend AttentionTileValueId with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileValueId with Eq::{not_equal, equal}

///|
pub extend AttentionTileValueId with Hash::{hash, hash_combine}

///|
pub extend AttentionTileExpression with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileExpression with Eq::{not_equal, equal}

///|
pub extend AttentionTileBinding with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileBinding with Eq::{not_equal, equal}

///|
pub extend AttentionTileProgramDigest with @moonbitlang/core/debug.Debug::{
  to_repr,
}

///|
pub extend AttentionTileProgramDigest with Eq::{not_equal, equal}

///|
pub extend AttentionTileProgramDigest with Hash::{hash, hash_combine}

///|
pub extend AttentionTileProgram with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileProgram with Eq::{not_equal, equal}

///|
pub extend AttentionTileIrRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileIrRule with Eq::{not_equal, equal}

///|
pub extend AttentionTileIrError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionTileIrError with Eq::{not_equal, equal}
