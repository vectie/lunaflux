// Offline original grouped BF16 einsum and UE8M0 output-B on literal attention.
// FP8 output-A startup conversion follows inference/convert.py exactly.
#include "original_tensor_io.h"

static int execute(const char *inference,const char *root,const char *input,const char *output,int rows,bool one_row) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned default_dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  Owned e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu")),fmt(PyUnicode_FromString("ue8m0"));
  if(!grad.p||!threads.p||!default_dtype.p||!u8.p||!e8.p||!fmt.p||
    PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0||PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0)return -1;
  Owned x(literal_bf16(torch.p,bf16.p,input,rows,32768));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned linear(PyObject_GetAttrString(model.p,"linear"));
  Owned einsum(PyObject_GetAttrString(torch.p,"einsum"));
  if(!x.p||!map||!open.p||!linear.p||!einsum.p)return -1;
  Owned wa(weight(open.p,map,root,"layers.0.attn.wo_a.weight"));
  Owned sa(weight(open.p,map,root,"layers.0.attn.wo_a.scale"));
  Owned wb(weight(open.p,map,root,"layers.0.attn.wo_b.weight"));
  Owned sb(weight(open.p,map,root,"layers.0.attn.wo_b.scale"));
  if(!wa.p||!sa.p||!wb.p||!sb.p||PyObject_SetAttrString(wb.p,"scale",sb.p)<0)return -1;
  // [8192,4096] -> [64,128,32,128]; block scale [64,32].
  Owned wf(PyObject_CallMethod(wa.p,"float",nullptr)),sf(PyObject_CallMethod(sa.p,"float",nullptr));
  Owned wv(wf.p?PyObject_CallMethod(wf.p,"view","iiii",64,128,32,128):nullptr);
  Owned sv(sf.p?PyObject_CallMethod(sf.p,"view","iiii",64,1,32,1):nullptr);
  Owned dequant(wv.p&&sv.p?PyObject_CallMethod(wv.p,"mul","O",sv.p):nullptr);
  Owned rounded(dequant.p?PyObject_CallMethod(dequant.p,"bfloat16",nullptr):nullptr);
  Owned grouped_weight(rounded.p?PyObject_CallMethod(rounded.p,"view","iii",8,1024,4096):nullptr);
  Owned grouped_input(PyObject_CallMethod(x.p,"view","iiii",1,rows,8,4096));
  Owned equation(PyUnicode_FromString("bsgd,grd->bsgr"));
  Owned args(grouped_weight.p&&grouped_input.p?PyTuple_Pack(3,equation.p,grouped_input.p,grouped_weight.p):nullptr);
  Owned oa,ob;
  if(one_row) {
    // The production capture advances one literal token at a time. Retain
    // that GEMM geometry instead of silently substituting a 144-row prefill.
    Owned allocation(PyObject_CallMethod(grouped_input.p,"new_empty","(iiii)",1,rows,8,1024));
    Owned output_allocation(PyObject_CallMethod(grouped_input.p,"new_empty","(iii)",1,rows,4096));
    if(!allocation.p||!output_allocation.p)return -1;
    oa.p=allocation.p;allocation.p=nullptr;
    ob.p=output_allocation.p;output_allocation.p=nullptr;
    for(int row=0;row<rows;++row) {
      Owned operand(PyObject_CallMethod(grouped_input.p,"narrow","iii",1,row,1));
      Owned operands(operand.p?PyTuple_Pack(3,equation.p,operand.p,grouped_weight.p):nullptr);
      Owned a(operands.p?PyObject_CallObject(einsum.p,operands.p):nullptr);
      Owned flat_row(a.p?PyObject_CallMethod(a.p,"flatten","i",2):nullptr);
      Owned b_operands(flat_row.p?PyTuple_Pack(2,flat_row.p,wb.p):nullptr);
      Owned b(b_operands.p?PyObject_CallObject(linear.p,b_operands.p):nullptr);
      Owned a_target(PyObject_CallMethod(oa.p,"narrow","iii",1,row,1));
      Owned b_target(PyObject_CallMethod(ob.p,"narrow","iii",1,row,1));
      Owned a_copy(a.p&&a_target.p?PyObject_CallMethod(a_target.p,"copy_","O",a.p):nullptr);
      Owned b_copy(b.p&&b_target.p?PyObject_CallMethod(b_target.p,"copy_","O",b.p):nullptr);
      if(!a_copy.p||!b_copy.p)return -1;
    }
  } else {
    oa.p=args.p?PyObject_CallObject(einsum.p,args.p):nullptr;
    Owned flat_batch(oa.p?PyObject_CallMethod(oa.p,"flatten","i",2):nullptr);
    Owned b_args(flat_batch.p?PyTuple_Pack(2,flat_batch.p,wb.p):nullptr);
    ob.p=b_args.p?PyObject_CallObject(linear.p,b_args.p):nullptr;
  }
  Owned flat(oa.p?PyObject_CallMethod(oa.p,"flatten","i",2):nullptr);
  if(!oa.p||!ob.p)return -1;
  // Attribution of the original row-parallel publication boundary. Execute
  // the two original K shards serially on one device, publish each BF16
  // partial, add in F32 and publish BF16 exactly as RowParallelLinear does.
  // This is not a distributed/NCCL or complete TP2-model qualification.
  Owned partial_sum;
  for(int rank=0;rank<2;++rank) {
    Owned input_view(PyObject_CallMethod(flat.p,"narrow","iii",2,rank*4096,4096));
    Owned operand(input_view.p?PyObject_CallMethod(input_view.p,"contiguous",nullptr):nullptr);
    Owned weight_view(PyObject_CallMethod(wb.p,"narrow","iii",1,rank*4096,4096));
    Owned matrix(weight_view.p?PyObject_CallMethod(weight_view.p,"contiguous",nullptr):nullptr);
    Owned scale_view(PyObject_CallMethod(sb.p,"narrow","iii",1,rank*32,32));
    Owned scales(scale_view.p?PyObject_CallMethod(scale_view.p,"contiguous",nullptr):nullptr);
    if(!operand.p||!matrix.p||!scales.p||PyObject_SetAttrString(matrix.p,"scale",scales.p)<0)return -1;
    Owned operands(PyTuple_Pack(2,operand.p,matrix.p));
    Owned result(PyObject_CallObject(linear.p,operands.p));
    if(!result.p||!save(result.p,u8.p,std::string(output)+"/output-b-partial-"+
      std::to_string(rank)+".bf16",int64_t(rows)*4096*2))return -1;
    Owned wide(PyObject_CallMethod(result.p,"float",nullptr));
    if(!wide.p)return -1;
    if(rank==0){partial_sum.p=wide.p;wide.p=nullptr;}
    else {
      Owned added(PyObject_CallMethod(partial_sum.p,"add_","O",wide.p));
      if(!added.p)return -1;
    }
  }
  Owned sharded(PyObject_CallMethod(partial_sum.p,"bfloat16",nullptr));
  if(!sharded.p||!save(sharded.p,u8.p,std::string(output)+"/output-b-sharded.bf16",
    int64_t(rows)*4096*2))return -1;
  if(!save(x.p,u8.p,std::string(output)+"/input.bf16",int64_t(rows)*32768*2)||
    !save(wa.p,u8.p,std::string(output)+"/wo_a.fp8",8192LL*4096)||
    !save(sa.p,u8.p,std::string(output)+"/wo_a.scale",64*32)||
    !save(wb.p,u8.p,std::string(output)+"/wo_b.fp8",4096LL*8192)||
    !save(sb.p,u8.p,std::string(output)+"/wo_b.scale",32*64)||
    !save(oa.p,u8.p,std::string(output)+"/output-a.bf16",int64_t(rows)*8192*2)||
    !save(ob.p,u8.p,std::string(output)+"/output-b.bf16",int64_t(rows)*4096*2))return -1;
  std::printf("original_grouped_output_complete rows=%d one_row_geometry=%s bf16_einsum=true original_fp8_output=true tp=1 serial_sharded_output_b=true sharded_control_geometry=batched full_model=false\n",rows,one_row?"true":"false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=6&&argc!=7)return 2;
  const bool one_row=argc==7&&std::string(argv[6])=="--one-row";
  if(argc==7&&!one_row)return 2;
  char *end=nullptr;long rows=std::strtol(argv[5],&end,10);
  if(end==argv[5]||!end||*end||rows<1||rows>144)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],static_cast<int>(rows),one_row);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
