///|
fn lower_luna_tile_cuda_mma(
  output : StringBuilder,
  program : LunaTileProgram,
  destination : LunaTileTensorId,
  left : LunaTileTensorId,
  right : LunaTileTensorId,
  accumulator : LunaTileTensorId,
  instruction_index : Int,
) -> Unit raise LunaTileError {
  let destination_tensor = require_luna_tile_tensor(
    program.tensors,
    destination,
  )
  let left_tensor = require_luna_tile_tensor(program.tensors, left)
  let right_tensor = require_luna_tile_tensor(program.tensors, right)
  let accumulator_tensor = require_luna_tile_tensor(
    program.tensors,
    accumulator,
  )
  let prefix = "m" + instruction_index.to_string()
  let m = prefix + "_m"
  let n = prefix + "_n"
  let k = prefix + "_k"
  let sum = prefix + "_sum"
  output.write_string("    for (long long " + m + " = 0; " + m + " < ")
  output.write_string(destination_tensor.shape[0].to_string())
  output.write_string("LL; ++" + m + ") {\n")
  output.write_string("    for (long long " + n + " = 0; " + n + " < ")
  output.write_string(destination_tensor.shape[1].to_string())
  output.write_string("LL; ++" + n + ") {\n")
  let accumulator_value = luna_tile_cuda_to_float(
    luna_tile_cuda_value(accumulator_tensor, [m, n]),
    accumulator_tensor.dtype,
  )
  output.write_string("      float " + sum + " = " + accumulator_value + ";\n")
  output.write_string("      for (long long " + k + " = 0; " + k + " < ")
  output.write_string(left_tensor.shape[1].to_string())
  output.write_string("LL; ++" + k + ") {\n")
  let left_value = luna_tile_cuda_to_float(
    luna_tile_cuda_value(left_tensor, [m, k]),
    left_tensor.dtype,
  )
  let right_value = luna_tile_cuda_to_float(
    luna_tile_cuda_value(right_tensor, [k, n]),
    right_tensor.dtype,
  )
  output.write_string(
    "        " +
    sum +
    " = fmaf(" +
    left_value +
    ", " +
    right_value +
    ", " +
    sum +
    ");\n",
  )
  output.write_string("      }\n")
  output.write_string(
    "      " +
    luna_tile_cuda_value(destination_tensor, [m, n]) +
    " = " +
    sum +
    ";\n",
  )
  output.write_string("    }\n    }\n")
}

///|
fn lower_luna_tile_cuda_reduce(
  output : StringBuilder,
  program : LunaTileProgram,
  kind : LunaTileReductionKind,
  destination : LunaTileTensorId,
  source : LunaTileTensorId,
  axis : Int,
  instruction_index : Int,
) -> Unit raise LunaTileError {
  let destination_tensor = require_luna_tile_tensor(
    program.tensors,
    destination,
  )
  let source_tensor = require_luna_tile_tensor(program.tensors, source)
  let prefix = "r" + instruction_index.to_string()
  let output_indices = luna_tile_cuda_open_loops(
    output,
    prefix,
    destination_tensor.shape,
    destination_tensor.shape.length(),
  )
  let reduction = prefix + "_reduce"
  let accumulator = prefix + "_acc"
  let initial = match kind {
    Sum => "0.0F"
    Maximum => "-3.402823466e+38F"
  }
  output.write_string("      float " + accumulator + " = " + initial + ";\n")
  output.write_string("      for (long long " + reduction + " = 0; ")
  output.write_string(reduction + " < " + source_tensor.shape[axis].to_string())
  output.write_string("LL; ++" + reduction + ") {\n")
  let source_indices = Array::makei(source_tensor.shape.length(), index => {
    if index == axis {
      reduction
    } else {
      output_indices[index]
    }
  })
  let value = luna_tile_cuda_to_float(
    luna_tile_cuda_value(source_tensor, source_indices),
    source_tensor.dtype,
  )
  match kind {
    Sum =>
      output.write_string(
        "        " + accumulator + " = " + accumulator + " + " + value + ";\n",
      )
    Maximum =>
      output.write_string(
        "        " +
        accumulator +
        " = fmaxf(" +
        accumulator +
        ", " +
        value +
        ");\n",
      )
  }
  output.write_string("      }\n")
  output.write_string("      ")
  output.write_string(luna_tile_cuda_value(destination_tensor, output_indices))
  output.write_string(" = ")
  output.write_string(
    luna_tile_cuda_from_float(accumulator, destination_tensor.dtype),
  )
  output.write_string(";\n")
  luna_tile_cuda_close_loops(output, destination_tensor.shape.length())
}

///|
fn lower_luna_tile_cuda_elementwise(
  output : StringBuilder,
  program : LunaTileProgram,
  kind : LunaTileElementwiseKind,
  destination : LunaTileTensorId,
  inputs : ReadOnlyArray[LunaTileTensorId],
  instruction_index : Int,
) -> Unit raise LunaTileError {
  let destination_tensor = require_luna_tile_tensor(
    program.tensors,
    destination,
  )
  let indices = luna_tile_cuda_open_loops(
    output,
    "e" + instruction_index.to_string(),
    destination_tensor.shape,
    destination_tensor.shape.length(),
  )
  let first_tensor = require_luna_tile_tensor(program.tensors, inputs[0])
  let first = luna_tile_cuda_to_float(
    luna_tile_cuda_value(first_tensor, indices),
    first_tensor.dtype,
  )
  let expression = match kind {
    Silu => "(" + first + " / (1.0F + expf(-" + first + ")))"
    Add | Multiply | Maximum => {
      let second_tensor = require_luna_tile_tensor(program.tensors, inputs[1])
      let second = luna_tile_cuda_to_float(
        luna_tile_cuda_value(second_tensor, indices),
        second_tensor.dtype,
      )
      match kind {
        Add => "(" + first + " + " + second + ")"
        Multiply => "(" + first + " * " + second + ")"
        Maximum => "fmaxf(" + first + ", " + second + ")"
        Silu => abort("exhaustive LunaTile elementwise kind")
      }
    }
  }
  output.write_string("      ")
  output.write_string(luna_tile_cuda_value(destination_tensor, indices))
  output.write_string(" = ")
  output.write_string(
    luna_tile_cuda_from_float(expression, destination_tensor.dtype),
  )
  output.write_string(";\n")
  luna_tile_cuda_close_loops(output, destination_tensor.shape.length())
}
