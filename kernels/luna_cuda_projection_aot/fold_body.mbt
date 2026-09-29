///|
priv enum CudaProjectionFragmentForm {
  Packed(
    products~ : ReadOnlyArray[(Int, String)],
    row_origin~ : String,
    row_owner~ : String,
    row_limit~ : String,
    column_owner~ : String,
    weight_displacement~ : String
  )
  HeadColumns(input_elements~ : Int)
}

///|
/// Closed backend binding of a physical fragment program. There is no public
/// source-body constructor: dependencies remain typed until terminal emission.
pub struct CudaProjectionFoldBody {
  priv program : @physical_tile_ir.FragmentProgram
  priv form : CudaProjectionFragmentForm
  priv storage : String
  priv slot_elements : Int
  priv consumers : Int?
}

///|
/// Bind the ordinary matrix consumer to the same checked operand storage used
/// by its producer. No independently reconstructed strip offsets or slot size.
fn projection_storage_fold_body(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  storage~ : String,
  consumers~ : Int,
) -> CudaProjectionFoldBody {
  let pipeline = physical.schedule
  let operands = physical.operands
  let (_, input_rows, width) = operands.region(0).unwrap().geometry()
  let (_, weight_rows, weight_width) = operands.region(1).unwrap().geometry()
  guard operands.region(2) is None &&
    width == pipeline.transfer_width() &&
    weight_width == width &&
    input_rows == pipeline.row_width() &&
    weight_rows == pipeline.output_tiles() * pipeline.tile_width() &&
    operands.allocation_elements() / operands.slot_elements() ==
    pipeline.stage_count() else {
    abort("matrix operand storage and fragment schedule disagree")
  }
  projection_packed_fold_body(physical, storage~, consumers~)
}

///|
fn projection_packed_fold_body(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
  storage~ : String,
  consumers? : Int,
  products? : Array[(Int, String)] = [],
  row_origin? : String = "row_begin",
  row_owner? : String = "consumer_row",
  row_limit? : String = "pipeline_rows",
  column_owner? : String = "consumer_column",
  weight_displacement? : String = "0",
) -> CudaProjectionFoldBody {
  let pipeline = physical.schedule
  let operands = physical.operands
  let products = if products.is_empty() { [(1, "c")] } else { products }
  let (_, input_rows, width) = operands.region(0).unwrap().geometry()
  guard projection_cuda_fragment_geometry(physical) &&
    width == pipeline.transfer_width() &&
    input_rows == pipeline.row_width() &&
    operands.allocation_elements() / operands.slot_elements() ==
    pipeline.stage_count() else {
    abort("operand storage and fragment schedule disagree")
  }
  let products = products.map(product => {
    let (region, accumulator) = product
    guard region > 0 else { abort("weight operand cannot alias input") }
    let (offset, _, _) = operands.region(region).unwrap().geometry()
    (offset, accumulator)
  })
  {
    program: physical.fragments,
    form: Packed(
      products=ReadOnlyArray::from_array(products),
      row_origin~,
      row_owner~,
      row_limit~,
      column_owner~,
      weight_displacement~,
    ),
    storage,
    slot_elements: operands.slot_elements(),
    consumers,
  }
}

///|
pub fn CudaProjectionFoldBody::ingress(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
) -> CudaProjectionFoldBody? {
  let pipeline = physical.schedule
  let operands = physical.operands
  let (rows, columns, _, _, _) = physical.fragments.geometry()
  guard operands.region(0) is Some(input) &&
    operands.region(1) is Some(weight) &&
    operands.region(2) is None else {
    return None
  }
  let (_, input_rows, width) = input.geometry()
  let (input_elements, weight_rows, _) = weight.geometry()
  let (_, vector, rotations, _) = operands.layout().geometry()
  let slot_elements = operands.slot_elements()
  guard projection_cuda_fragment_geometry(physical) &&
    pipeline.tile_width() == 16 &&
    input_rows == 16 &&
    width == pipeline.transfer_width() &&
    input_rows == pipeline.row_width() &&
    weight_rows == pipeline.output_tiles() * 16 &&
    vector == 8 &&
    rotations == 1 &&
    operands.allocation_elements() / slot_elements == pipeline.stage_count() &&
    rows == 1 &&
    columns > 0 &&
    columns <= pipeline.output_tiles() else {
    return None
  }
  Some({
    program: physical.fragments,
    form: HeadColumns(input_elements~),
    storage: "ingress_operands",
    slot_elements,
    consumers: None,
  })
}

///|
/// Instruction geometry belongs to this backend binding, not the generic IR.
fn projection_cuda_fragment_geometry(
  physical : @luna_projection_tile_compiler.ProjectionPhysicalPipeline,
) -> Bool {
  physical.owners_per_group == 32 &&
  physical.operands.layout().geometry().1 == 8 &&
  physical.fragments.geometry().3 == 16
}

///|
fn CudaProjectionFoldBody::source(self : CudaProjectionFoldBody) -> String {
  let fragment = match self.form {
    Packed(
      products~,
      row_origin~,
      row_owner~,
      row_limit~,
      column_owner~,
      weight_displacement~
    ) =>
      projection_fragment_pipeline_source(
        self.program,
        products,
        row_origin~,
        row_owner~,
        row_limit~,
        column_owner~,
        weight_displacement~,
      )
    HeadColumns(input_elements~) =>
      projection_head_fragment_source(self.program, input_elements)
  }
  let prefix = "const __nv_bfloat16 *current = \{self.storage} + stage * \{self.slot_elements};\n"
  match self.consumers {
    Some(count) => "if (warp < \{count}) {\n" + prefix + fragment + "\n}"
    None => prefix + fragment
  }
}
