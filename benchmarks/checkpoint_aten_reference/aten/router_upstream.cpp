// Offline original Gate on literal FFN inputs. No expert or full model load.
#include "original_tensor_io.h"

static int execute(const char *inference,const char *root,const char *input,
                   const char *ids_path,const char *out,int layer) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors")),functional(PyImport_ImportModule("torch.nn.functional"));
  if(!torch.p||!model.p||!safe.p||!functional.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned i32(PyObject_GetAttrString(torch.p,"int32")),i64(PyObject_GetAttrString(torch.p,"int64"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned literal(literal_bf16(torch.p,bf16.p,input,144,4096));
  Owned x(literal.p?PyObject_CallMethod(literal.p,"view","ii",144,4096):nullptr);
  Owned ids0(literal_tensor(torch.p,i32.p,ids_path,144,1,4));
  Owned ids1(ids0.p?PyObject_CallMethod(ids0.p,"flatten",nullptr):nullptr);
  Owned ids(ids1.p?PyObject_CallMethod(ids1.p,"to","O",i64.p):nullptr);
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned cls(PyObject_GetAttrString(model.p,"ModelArgs")),empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:s,s:d}",
    "dim",4096,"n_routed_experts",256,"n_activated_experts",6,"n_hash_layers",3,
    "vocab_size",129280,"score_func","sqrtsoftplus","route_scale",1.5));
  Owned config(PyObject_Call(cls.p,empty.p,options.p)),gate_cls(PyObject_GetAttrString(model.p,"Gate"));
  Owned args(Py_BuildValue("iO",layer,config.p));
  Owned gate(PyObject_CallObject(gate_cls.p,args.p));
  Owned gpu(gate.p?PyObject_CallMethod(gate.p,"to","s","cuda"):nullptr);
  if(!grad.p||!threads.p||!dtype.p||!x.p||!ids.p||!map||!open.p||!gpu.p)return -1;
  const std::string prefix="layers."+std::to_string(layer)+".ffn.gate.";
  for(const char *name:{"weight",layer<3?"tid2eid":"bias"}) {
    Owned target(PyObject_GetAttrString(gpu.p,name));
    Owned tensor(weight(open.p,map,root,(prefix+name).c_str()));
    Owned copied(target.p&&tensor.p?PyObject_CallMethod(target.p,"copy_","O",tensor.p):nullptr);
    const int64_t bytes=std::string(name)=="weight"?256*4096*2:
      layer<3?129280*6*4:256*4;
    const std::string suffix=std::string(name)=="weight"?"weight.bf16":layer<3?"table.i32":"bias.f32";
    if(!copied.p||!save(target.p,u8.p,std::string(out)+"/"+suffix,bytes))return -1;
  }
  Owned operands(PyTuple_Pack(2,x.p,ids.p)),routed(PyObject_CallObject(gpu.p,operands.p));
  Owned w(PyObject_GetAttrString(gpu.p,"weight")),xf(PyObject_CallMethod(x.p,"float",nullptr));
  Owned wf(w.p?PyObject_CallMethod(w.p,"float",nullptr):nullptr);
  Owned linear(PyObject_GetAttrString(model.p,"linear"));
  Owned linear_args(xf.p&&wf.p?PyTuple_Pack(2,xf.p,wf.p):nullptr);
  Owned logits(linear_args.p?PyObject_CallObject(linear.p,linear_args.p):nullptr);
  Owned softplus(logits.p?PyObject_CallMethod(functional.p,"softplus","O",logits.p):nullptr);
  Owned scores(softplus.p?PyObject_CallMethod(softplus.p,"sqrt",nullptr):nullptr);
  if(!routed.p||!PyTuple_Check(routed.p)||PyTuple_Size(routed.p)!=2||!scores.p)return -1;
  Owned selected(PyObject_CallMethod(PyTuple_GetItem(routed.p,1),"to","O",i32.p));
  if(!save(logits.p,u8.p,std::string(out)+"/logits.f32",144*256*4)||
     !save(scores.p,u8.p,std::string(out)+"/scores.f32",144*256*4)||
     !save(selected.p,u8.p,std::string(out)+"/indices.i32",144*6*4)||
     !save(PyTuple_GetItem(routed.p,0),u8.p,std::string(out)+"/weights.f32",144*6*4))return -1;
  std::printf("original_router_complete layer=%d rows=144 inputs=literal_original_ffn full_model=false\n",layer);
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=7)return 2;
  const int layer=std::atoi(argv[6]);if(layer!=0&&layer!=3)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],argv[5],layer);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
