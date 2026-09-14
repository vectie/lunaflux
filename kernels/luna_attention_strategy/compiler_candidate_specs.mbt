///|
priv struct AttentionCandidateSpec {
  stable_id : Int
  phase : AttentionExecutionPhase
  compute : AttentionTileCompute
  memory : AttentionTileMemory
  query_rows : Int
  key_value_rows : Int
  subgroups_per_block : Int
  pipeline_stages : Int
  vector_bytes : Int
  minimum_query_rows : Int
  minimum_context_tokens : Int
}

///|
/// Realized scalar/subgroup domains, not calibrated cost coefficients.
fn attention_candidate_specs() -> Array[AttentionCandidateSpec] {
  let specs : Array[AttentionCandidateSpec] = []
  for phase in [Prefill, Decode] {
    specs.push({
      stable_id: if phase == Prefill {
        100
      } else {
        400
      },
      phase,
      compute: AttentionScalarDot,
      memory: AttentionDirectGlobal,
      query_rows: 1,
      key_value_rows: 16,
      subgroups_per_block: 1,
      pipeline_stages: 1,
      vector_bytes: 4,
      minimum_query_rows: 1,
      minimum_context_tokens: 1,
    })
    let queries = if phase == Prefill { [4, 8, 16] } else { [1] }
    for index, query in queries {
      for asynchronous in [false, true] {
        // A four-query async realization has not been implemented.
        if phase == Prefill && query == 4 && asynchronous {
          continue
        }
        let groups = if phase == Decode { 4 } else { Int::max(4, query / 2) }
        let keys = if phase == Decode {
          if asynchronous {
            128
          } else {
            64
          }
        } else if asynchronous {
          query * 8
        } else {
          Int::min(64, query * 8)
        }
        specs.push({
          stable_id: (if phase == Prefill { 200 + index * 10 } else { 410 }) +
          (if asynchronous { 1 } else { 0 }),
          phase,
          compute: AttentionSubgroupDot,
          memory: if asynchronous {
            AttentionAsyncStagedShared
          } else {
            AttentionStagedShared
          },
          query_rows: query,
          key_value_rows: keys,
          subgroups_per_block: groups,
          pipeline_stages: if asynchronous {
            3
          } else if phase == Decode {
            2
          } else {
            1
          },
          vector_bytes: if asynchronous {
            16
          } else {
            8
          },
          minimum_query_rows: query,
          minimum_context_tokens: if phase == Prefill && query == 4 {
            16
          } else {
            32
          },
        })
      }
    }
    if phase == Prefill {
      specs.append(matrix_dot_specs())
    }
  }
  specs
}

///|
/// Matrix-QK with subgroup PV has a different realized transfer domain from
/// matrix-QK/PV. It must not inherit an unsupported Cartesian product.
fn matrix_dot_specs() -> Array[AttentionCandidateSpec] {
  let specs : Array[AttentionCandidateSpec] = []
  for groups in [4, 8] {
    for variant in [0, 1] {
      let asynchronous = groups == 8 && variant == 1
      specs.push({
        stable_id: (if groups == 4 { 290 } else { 300 }) + variant,
        phase: Prefill,
        compute: AttentionMatrixDot,
        memory: if asynchronous {
          AttentionAsyncStagedShared
        } else {
          AttentionStagedShared
        },
        query_rows: 16,
        key_value_rows: groups * 8 * (variant + 1),
        subgroups_per_block: groups,
        pipeline_stages: if asynchronous {
          3
        } else {
          1
        },
        vector_bytes: 16,
        minimum_query_rows: 16,
        minimum_context_tokens: 32,
      })
    }
  }
  specs
}
