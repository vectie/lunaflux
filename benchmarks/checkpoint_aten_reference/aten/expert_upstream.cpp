// Offline original hash router and its six actual routed FP4 experts, row zero.
// Linear, FP8 activation quantization and FP4 matrix kernels remain original.
#include "original_tensor_io.h"

static bool copy_parameter(PyObject *object,PyObject *open,PyObject *map,const char *root,
                           const char *name,const std::string &source) {
  Owned target(PyObject_GetAttrString(object,name));
  Owned tensor(weight(open,map,root,source.c_str()));
  Owned copied(target.p&&tensor.p?PyObject_CallMethod(target.p,"copy_","O",tensor.p):nullptr);
  return copied.p!=nullptr;
}

static int execute(const char *inference,const char *root,const char *input,const char *out,int token) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned f32(PyObject_GetAttrString(torch.p,"float32")),i64(PyObject_GetAttrString(torch.p,"int64"));
  Owned fp4(PyObject_GetAttrString(torch.p,"float4_e2m1fn_x2"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu")),fmt(PyUnicode_FromString("ue8m0"));
  if(!grad.p||!threads.p||!bf16.p||!u8.p||!f32.p||!i64.p||!fp4.p||!dtype.p||!e8.p||!fmt.p||
    PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0||PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0)return -1;
  Owned literal(literal_bf16(torch.p,bf16.p,input,1,4096));
  Owned x(literal.p?PyObject_CallMethod(literal.p,"view","ii",1,4096):nullptr);
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned args_cls(PyObject_GetAttrString(model.p,"ModelArgs")),empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:s,s:d}",
    "dim",4096,"n_routed_experts",256,"n_activated_experts",6,"n_hash_layers",3,
    "vocab_size",129280,"score_func","sqrtsoftplus","route_scale",1.5));
  Owned config(PyObject_Call(args_cls.p,empty.p,options.p));
  Owned gate_cls(PyObject_GetAttrString(model.p,"Gate"));
  Owned gate_args(Py_BuildValue("iO",0,config.p));
  Owned gate(gate_cls.p?PyObject_CallObject(gate_cls.p,gate_args.p):nullptr);
  Owned gpu_gate(gate.p?PyObject_CallMethod(gate.p,"to","s","cuda"):nullptr);
  if(!x.p||!map||!open.p||!gpu_gate.p||
     !copy_parameter(gpu_gate.p,open.p,map,root,"weight","layers.0.ffn.gate.weight")||
     !copy_parameter(gpu_gate.p,open.p,map,root,"tid2eid","layers.0.ffn.gate.tid2eid"))return -1;
  Owned ids_list(Py_BuildValue("[i]",token)),factory(PyObject_GetAttrString(torch.p,"tensor"));
  Owned ids_args(PyTuple_Pack(1,ids_list.p)),ids_kw(Py_BuildValue("{s:O,s:s}","dtype",i64.p,"device","cuda"));
  Owned ids(PyObject_Call(factory.p,ids_args.p,ids_kw.p));
  Owned routing_args(PyTuple_Pack(2,x.p,ids.p));
  Owned routing(PyObject_CallObject(gpu_gate.p,routing_args.p));
  if(!routing.p||!PyTuple_Check(routing.p)||PyTuple_Size(routing.p)!=2)return -1;
  auto *scores=PyTuple_GetItem(routing.p,0),*selected=PyTuple_GetItem(routing.p,1);
  Owned cpu_indices(PyObject_CallMethod(selected,"cpu",nullptr));
  Owned list(cpu_indices.p?PyObject_CallMethod(cpu_indices.p,"tolist",nullptr):nullptr);
  Owned functional(PyImport_ImportModule("torch.nn.functional")),linear(PyObject_GetAttrString(model.p,"linear"));
  Owned expert_cls(PyObject_GetAttrString(model.p,"Expert"));
  if(!list.p||!functional.p||!linear.p||!expert_cls.p||
     !save(x.p,u8.p,std::string(out)+"/input.bf16",8192)||
     !save(scores,u8.p,std::string(out)+"/scores.f32",24))return -1;
  for(int slot=0;slot<6;++slot) {
    const int id=static_cast<int>(PyLong_AsLong(PyList_GetItem(PyList_GetItem(list.p,0),slot)));
    std::printf("original_expert_slot=%d id=%d token=%d\n",slot,id,token);
    Owned ctor_args(Py_BuildValue("ii",4096,2048));
    Owned ctor_kw(Py_BuildValue("{s:O,s:d}","dtype",fp4.p,"swiglu_limit",10.0));
    Owned expert(PyObject_Call(expert_cls.p,ctor_args.p,ctor_kw.p));
    Owned gpu(expert.p?PyObject_CallMethod(expert.p,"to","s","cuda"):nullptr);
    if(!gpu.p)return -1;
    for(int stage=0;stage<3;++stage) {
      const char *name=stage==0?"w1":stage==1?"w3":"w2";
      const std::string prefix="layers.0.ffn.experts."+std::to_string(id)+"."+name;
      Owned matrix(PyObject_GetAttrString(gpu.p,name));
      Owned target(matrix.p?PyObject_GetAttrString(matrix.p,"weight"):nullptr);
      Owned target_scale(matrix.p?PyObject_GetAttrString(matrix.p,"scale"):nullptr);
      Owned payload(weight(open.p,map,root,(prefix+".weight").c_str()));
      Owned scale(weight(open.p,map,root,(prefix+".scale").c_str()));
      // Shell dtypes do not implement typed copy on every installed torch build.
      // Copy their literal byte representations, preserving the checkpoint bits.
      Owned target_bytes(target.p?PyObject_CallMethod(target.p,"view","O",u8.p):nullptr);
      Owned payload_bytes(payload.p?PyObject_CallMethod(payload.p,"view","O",u8.p):nullptr);
      Owned scale_bytes(target_scale.p?PyObject_CallMethod(target_scale.p,"view","O",u8.p):nullptr);
      Owned source_scale_bytes(scale.p?PyObject_CallMethod(scale.p,"view","O",u8.p):nullptr);
      Owned copied(target_bytes.p&&payload_bytes.p?PyObject_CallMethod(target_bytes.p,"copy_","O",payload_bytes.p):nullptr);
      Owned copied_scale(scale_bytes.p&&source_scale_bytes.p?PyObject_CallMethod(scale_bytes.p,"copy_","O",source_scale_bytes.p):nullptr);
      if(!copied.p||!copied_scale.p||PyObject_SetAttrString(target.p,"scale",target_scale.p)<0)return -1;
      const int width=stage==2?2048:4096,rows=stage==2?4096:2048;
      const std::string path=std::string(out)+"/slot"+std::to_string(slot)+"-w"+std::to_string(stage);
      if(!save(payload.p,u8.p,path+".fp4",int64_t(rows)*width/2)||
         !save(scale.p,u8.p,path+".scale",int64_t(rows)*width/32))return -1;
    }
    Owned score(PyObject_CallMethod(scores,"narrow","iii",1,slot,1));
    Owned operands(PyTuple_Pack(2,x.p,score.p));
    Owned result(PyObject_CallObject(gpu.p,operands.p));
    // Capture the exact original down input independently of its matrix kernel.
    Owned w1(PyObject_GetAttrString(gpu.p,"w1")),w3(PyObject_GetAttrString(gpu.p,"w3"));
    Owned one(PyTuple_Pack(1,x.p)),g0(PyObject_CallObject(w1.p,one.p)),u0(PyObject_CallObject(w3.p,one.p));
    Owned gf(g0.p?PyObject_CallMethod(g0.p,"float",nullptr):nullptr),uf(u0.p?PyObject_CallMethod(u0.p,"float",nullptr):nullptr);
    Owned clamped_gate(gf.p?PyObject_CallMethod(gf.p,"clamp_max","d",10.0):nullptr);
    Owned clamped_up(uf.p?PyObject_CallMethod(uf.p,"clamp","dd",-10.0,10.0):nullptr);
    Owned silu(clamped_gate.p?PyObject_CallMethod(functional.p,"silu","O",clamped_gate.p):nullptr);
    Owned product(silu.p&&clamped_up.p?PyObject_CallMethod(silu.p,"mul","O",clamped_up.p):nullptr);
    Owned weighted(product.p&&score.p?PyObject_CallMethod(product.p,"mul","O",score.p):nullptr);
    Owned rounded(weighted.p?PyObject_CallMethod(weighted.p,"bfloat16",nullptr):nullptr);
    const std::string stem=std::string(out)+"/slot"+std::to_string(slot);
    if(!result.p||!rounded.p||!save(result.p,u8.p,stem+"-output.bf16",8192)||
       !save(rounded.p,u8.p,stem+"-activation.bf16",4096))return -1;
  }
  std::puts("original_experts_complete actual_hash_routes=true route_before_down=true full_model=false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=6)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],std::atoi(argv[5]));
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
