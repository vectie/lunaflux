// Offline literal shared-expert operands. Calls the unmodified original Expert
// and its FP8 linears; no model implementation enters production.
#include "original_tensor_io.h"

static int execute(const char *inference,const char *root,const char *input,const char *out) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned fp8(PyObject_GetAttrString(torch.p,"float8_e4m3fn")),e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu"));
  Owned fmt(PyUnicode_FromString("ue8m0"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  if(!grad.p||!threads.p||!dtype.p||!fp8.p||!e8.p||!fmt.p||
     PyObject_SetAttrString(model.p,"default_dtype",fp8.p)<0||
     PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0||
     PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0)return -1;
  Owned literal(literal_bf16(torch.p,bf16.p,input,1,4096));
  Owned x(literal.p?PyObject_CallMethod(literal.p,"view","ii",1,4096):nullptr);
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned cls(PyObject_GetAttrString(model.p,"Expert")),args(Py_BuildValue("ii",4096,2048));
  Owned kw(Py_BuildValue("{s:O,s:d}","dtype",fp8.p,"swiglu_limit",10.0));
  Owned expert(cls.p?PyObject_Call(cls.p,args.p,kw.p):nullptr);
  Owned gpu(expert.p?PyObject_CallMethod(expert.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!map||!open.p||!x.p)return -1;
  for(const char *name:{"w1","w3","w2"}) {
    Owned matrix(PyObject_GetAttrString(gpu.p,name));
    Owned target(matrix.p?PyObject_GetAttrString(matrix.p,"weight"):nullptr);
    Owned target_scale(matrix.p?PyObject_GetAttrString(matrix.p,"scale"):nullptr);
    const std::string key="layers.0.ffn.shared_experts."+std::string(name);
    Owned payload(weight(open.p,map,root,(key+".weight").c_str()));
    Owned scale(weight(open.p,map,root,(key+".scale").c_str()));
    Owned a(target.p?PyObject_CallMethod(target.p,"view","O",u8.p):nullptr);
    Owned b(payload.p?PyObject_CallMethod(payload.p,"view","O",u8.p):nullptr);
    Owned sa(target_scale.p?PyObject_CallMethod(target_scale.p,"view","O",u8.p):nullptr);
    Owned sb(scale.p?PyObject_CallMethod(scale.p,"view","O",u8.p):nullptr);
    Owned copied(a.p&&b.p?PyObject_CallMethod(a.p,"copy_","O",b.p):nullptr);
    Owned copied_scale(sa.p&&sb.p?PyObject_CallMethod(sa.p,"copy_","O",sb.p):nullptr);
    if(!copied.p||!copied_scale.p||PyObject_SetAttrString(target.p,"scale",target_scale.p)<0||
       !save(payload.p,u8.p,std::string(out)+"/"+name+".fp8",8388608)||
       !save(scale.p,u8.p,std::string(out)+"/"+name+".scale",512))return -1;
  }
  Owned call(PyTuple_Pack(1,x.p)),result(PyObject_CallObject(gpu.p,call.p));
  Owned gate(PyObject_GetAttrString(gpu.p,"w1")),up(PyObject_GetAttrString(gpu.p,"w3"));
  Owned g(PyObject_CallObject(gate.p,call.p)),u(PyObject_CallObject(up.p,call.p));
  Owned gf(g.p?PyObject_CallMethod(g.p,"float",nullptr):nullptr),uf(u.p?PyObject_CallMethod(u.p,"float",nullptr):nullptr);
  Owned gc(gf.p?PyObject_CallMethod(gf.p,"clamp_max","d",10.0):nullptr);
  Owned uc(uf.p?PyObject_CallMethod(uf.p,"clamp","dd",-10.0,10.0):nullptr);
  Owned functional(PyImport_ImportModule("torch.nn.functional"));
  Owned activated(gc.p?PyObject_CallMethod(functional.p,"silu","O",gc.p):nullptr);
  Owned product(activated.p&&uc.p?PyObject_CallMethod(activated.p,"mul","O",uc.p):nullptr);
  Owned rounded(product.p?PyObject_CallMethod(product.p,"bfloat16",nullptr):nullptr);
  if(!g.p||!u.p||!result.p||!rounded.p||!save(x.p,u8.p,std::string(out)+"/input.bf16",8192)||
     !save(g.p,u8.p,std::string(out)+"/gate.bf16",4096)||
     !save(u.p,u8.p,std::string(out)+"/up.bf16",4096)||
     !save(rounded.p,u8.p,std::string(out)+"/activation.bf16",4096)||
     !save(result.p,u8.p,std::string(out)+"/output.bf16",8192))return -1;
  // Stateless observations of the installed original quantizer and down
  // kernel. The Expert call above is untouched; no observation is fed back.
  Owned kernel(PyImport_ImportModule("kernel"));
  Owned quantizer(kernel.p?PyObject_GetAttrString(kernel.p,"act_quant"):nullptr);
  Owned quant_kw(Py_BuildValue("{s:O,s:O}","scale_fmt",fmt.p,"scale_dtype",e8.p));
  Owned block(PyLong_FromLong(128));
  Owned qx_args(PyTuple_Pack(2,x.p,block.p)),qa_args(PyTuple_Pack(2,rounded.p,block.p));
  Owned qx(quantizer.p?PyObject_Call(quantizer.p,qx_args.p,quant_kw.p):nullptr);
  Owned qa(quantizer.p?PyObject_Call(quantizer.p,qa_args.p,quant_kw.p):nullptr);
  if(!qx.p||!qa.p)return -1;
  auto *qx_data=PyTuple_GetItem(qx.p,0),*qx_scale=PyTuple_GetItem(qx.p,1);
  auto *qa_data=PyTuple_GetItem(qa.p,0),*qa_scale=PyTuple_GetItem(qa.p,1);
  if(!qx_data||!qx_scale||!qa_data||!qa_scale||
     !save(qx_data,u8.p,std::string(out)+"/input-quantized.fp8",4096)||
     !save(qx_scale,u8.p,std::string(out)+"/input-quantized.scale",32)||
     !save(qa_data,u8.p,std::string(out)+"/activation-quantized.fp8",2048)||
     !save(qa_scale,u8.p,std::string(out)+"/activation-quantized.scale",16))return -1;
  Owned down(PyObject_GetAttrString(gpu.p,"w2")),down_args(PyTuple_Pack(1,rounded.p));
  Owned matched(down.p?PyObject_CallObject(down.p,down_args.p):nullptr);
  if(!matched.p||!save(matched.p,u8.p,std::string(out)+"/matched-output.bf16",8192))return -1;
  // The same installed GEMM algorithm supports a float32 terminal. Retain its
  // pre-BF16 result as an arithmetic frontier, not as a replacement reference.
  Owned factory(kernel.p?PyObject_GetAttrString(kernel.p,"fp8_gemm_kernel"):nullptr);
  Owned geometry(Py_BuildValue("ii",4096,2048));
  Owned options(Py_BuildValue("{s:s,s:s}","out_dtype","float32","scale_dtype","float8_e8m0fnu"));
  Owned original_gemm(factory.p?PyObject_Call(factory.p,geometry.p,options.p):nullptr);
  Owned f32(PyObject_GetAttrString(torch.p,"float32"));
  Owned empty(PyObject_GetAttrString(torch.p,"empty")),shape(Py_BuildValue("(ii)",1,4096));
  Owned empty_args(PyTuple_Pack(1,shape.p)),empty_kw(Py_BuildValue("{s:O,s:s}","dtype",f32.p,"device","cuda"));
  Owned frontier(empty.p?PyObject_Call(empty.p,empty_args.p,empty_kw.p):nullptr);
  Owned dw(down.p?PyObject_GetAttrString(down.p,"weight"):nullptr),ds(down.p?PyObject_GetAttrString(down.p,"scale"):nullptr);
  Owned gemm_args(frontier.p&&dw.p&&ds.p?PyTuple_Pack(5,qa_data,dw.p,frontier.p,qa_scale,ds.p):nullptr);
  Owned observed(original_gemm.p&&gemm_args.p?PyObject_CallObject(original_gemm.p,gemm_args.p):nullptr);
  if(!observed.p||!save(frontier.p,u8.p,std::string(out)+"/matched-output.f32",16384))return -1;
  Owned terminal(frontier.p?PyObject_CallMethod(frontier.p,"bfloat16",nullptr):nullptr);
  if(!terminal.p||!save(terminal.p,u8.p,std::string(out)+"/f32-terminal.bf16",8192))return -1;
  // Observe each original unscaled 128-column MMA group independently. Ones
  // remove scale application for this offline frontier only. The Expert and
  // preceding scaled outputs remain their unmodified installed computations.
  Owned block_geometry(Py_BuildValue("ii",4096,128));
  Owned block_options(Py_BuildValue("{s:s,s:s}","out_dtype","float32","scale_dtype","float32"));
  Owned original_block(factory.p?PyObject_Call(factory.p,block_geometry.p,block_options.p):nullptr);
  Owned ones(PyObject_GetAttrString(torch.p,"ones"));
  Owned a_shape(Py_BuildValue("(ii)",1,1)),b_shape(Py_BuildValue("(ii)",32,1));
  Owned a_args(PyTuple_Pack(1,a_shape.p)),b_args(PyTuple_Pack(1,b_shape.p));
  Owned a_scale(ones.p?PyObject_Call(ones.p,a_args.p,empty_kw.p):nullptr);
  Owned b_scale(ones.p?PyObject_Call(ones.p,b_args.p,empty_kw.p):nullptr);
  if(!original_block.p||!a_scale.p||!b_scale.p)return -1;
  for(int group=0;group<16;++group) {
    Owned a_slice(PyObject_CallMethod(qa_data,"narrow","iii",1,group*128,128));
    Owned b_slice(PyObject_CallMethod(dw.p,"narrow","iii",1,group*128,128));
    Owned a(a_slice.p?PyObject_CallMethod(a_slice.p,"contiguous",nullptr):nullptr);
    Owned b(b_slice.p?PyObject_CallMethod(b_slice.p,"contiguous",nullptr):nullptr);
    Owned args(a.p&&b.p?PyTuple_Pack(5,a.p,b.p,frontier.p,a_scale.p,b_scale.p):nullptr);
    Owned result(args.p?PyObject_CallObject(original_block.p,args.p):nullptr);
    if(!result.p||!save(frontier.p,u8.p,std::string(out)+"/down-unscaled-block-"+std::to_string(group)+".f32",16384))return -1;
  }
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=5)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
