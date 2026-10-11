// Exact caption encoding -> original condition_proj -> original two refiners.
// Only checkpoint layout and literal tensor I/O are supplied by this adapter.
#include "original_h3_loading.h"
#include <cstdlib>

static int execute(const char *root,const char *input,const char *out,int rows,const char *matched) {
  Owned torch(PyImport_ImportModule("torch")),safe(PyImport_ImportModule("safetensors"));
  Owned distributed(PyImport_ImportModule("vllm.distributed")),config_module(PyImport_ImportModule("vllm.config"));
  Owned config_cls(config_module.p?PyObject_GetAttrString(config_module.p,"VllmConfig"):nullptr);
  Owned config(config_cls.p?PyObject_CallNoArgs(config_cls.p):nullptr);
  Owned scope(config.p?PyObject_CallMethod(config_module.p,"set_current_vllm_config","O",config.p):nullptr);
  Owned entered(scope.p?PyObject_CallMethod(scope.p,"__enter__",nullptr):nullptr);
  if(!torch.p||!safe.p||!distributed.p||!entered.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned i32(PyObject_GetAttrString(torch.p,"int32"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned init(PyObject_GetAttrString(distributed.p,"init_distributed_environment"));
  Owned empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i,s:i,s:s,s:s}","world_size",1,"rank",0,"local_rank",0,
    "distributed_init_method","file:///work/rendezvous","backend","gloo"));
  Owned world(init.p?PyObject_Call(init.p,empty.p,options.p):nullptr);
  Owned parallel(PyObject_CallMethod(distributed.p,"initialize_model_parallel","ii",1,1));
  if(!grad.p||!threads.p||!dtype.p||!world.p||!parallel.p)return -1;
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer"));
  Owned metadata(json_file(std::string(root)+"/config.json"));
  Owned arch_cls(module.p?PyObject_GetAttrString(module.p,"MiniMaxH3DiTArchConfig"):nullptr);
  Owned arch(arch_cls.p&&metadata.p?PyObject_CallMethod(arch_cls.p,"from_mapping","O",metadata.p):nullptr);
  Owned index(json_file(std::string(root)+"/diffusion_pytorch_model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned projection_cls(module.p?PyObject_GetAttrString(module.p,"ColumnParallelLinear"):nullptr);
  Owned dimensions(Py_BuildValue("(ii)",5120,5376));
  Owned kw(Py_BuildValue("{s:O,s:O,s:O,s:O}","bias",Py_True,"gather_output",Py_True,
    "params_dtype",bf16.p,"quant_config",Py_None));
  Owned projection(projection_cls.p?PyObject_Call(projection_cls.p,dimensions.p,kw.p):nullptr);
  Owned gpu(projection.p?PyObject_CallMethod(projection.p,"to","s","cuda"):nullptr);
  if(!gpu.p||!map||!open.p||!arch.p)return -1;
  if(!h3_parameter(gpu.p,open.p,map,root,"weight","context_embedder.weight")||
     !h3_parameter(gpu.p,open.p,map,root,"bias","context_embedder.bias"))return -1;
  Owned literal(literal_bf16(torch.p,bf16.p,input,rows,5120));
  Owned flat(literal.p?PyObject_CallMethod(literal.p,"view","ii",rows,5120):nullptr);
  Owned args(flat.p?PyTuple_Pack(1,flat.p):nullptr);
  Owned pair(args.p?PyObject_CallObject(gpu.p,args.p):nullptr);
  auto *projected=pair.p?PyTuple_GetItem(pair.p,0):nullptr;
  if(!projected||!save(projected,u8.p,std::string(out)+"/context.bf16",int64_t(rows)*5376*2))return -1;
  Owned refiner_cls(PyObject_GetAttrString(module.p,"MiniMaxH3TokenRefiner"));
  Owned ctor(PyTuple_Pack(2,arch.p,Py_None));
  Owned refiner(refiner_cls.p?PyObject_CallObject(refiner_cls.p,ctor.p):nullptr);
  Owned refiner_gpu(refiner.p?PyObject_CallMethod(refiner.p,"to","s","cuda"):nullptr);
  Owned blocks(refiner_gpu.p?PyObject_GetAttrString(refiner_gpu.p,"blocks"):nullptr);
  Owned boundaries(Py_BuildValue("[ii]",0,rows));
  Owned tensor_cls(PyObject_GetAttrString(torch.p,"tensor"));
  Owned bargs(PyTuple_Pack(1,boundaries.p));
  Owned bkw(Py_BuildValue("{s:O,s:s}","dtype",i32.p,"device","cuda"));
  Owned docs(tensor_cls.p?PyObject_Call(tensor_cls.p,bargs.p,bkw.p):nullptr);
  Owned attention_kw(docs.p?Py_BuildValue("{s:O,s:i}","cu_seqlens",docs.p,"max_seqlen",rows):nullptr);
  if(!blocks.p||!attention_kw.p)return -1;
  Owned state(Py_NewRef(projected));
  for(int layer=0;layer<2;++layer) {
    Owned key(PyLong_FromLong(layer));
    Owned block(PyObject_GetItem(blocks.p,key.p));
    if(!block.p||!h3_attention_mlp(block.p,open.p,map,root,
       "token_refiner.refiner_blocks."+std::to_string(layer)+"."))return -1;
    // An explicit intervention supplies each literal native predecessor. Do
    // not propagate a reference output and call it a matched local operator.
    Owned prior(matched?literal_bf16(torch.p,bf16.p,
      (std::string(matched)+"/"+(layer?"refiner-0.bf16":"context.bf16")).c_str(),rows,5376):Py_NewRef(state.p));
    Owned flat_prior(prior.p?PyObject_CallMethod(prior.p,"view","ii",rows,5376):nullptr);
    Owned operands(flat_prior.p?PyTuple_Pack(1,flat_prior.p):nullptr);
    if(!operands.p)return -1;
    Owned next(PyObject_Call(block.p,operands.p,attention_kw.p));
    if(!next.p||!save(next.p,u8.p,std::string(out)+"/refiner-"+std::to_string(layer)+".bf16",int64_t(rows)*5376*2))return -1;
    Py_SETREF(state.p,Py_NewRef(next.p));
  }
  Owned norm(PyObject_GetAttrString(refiner_gpu.p,"final_norm"));
  if(!norm.p||!h3_parameter(norm.p,open.p,map,root,"weight","token_refiner.final_norm.weight"))return -1;
  Owned prior(matched?literal_bf16(torch.p,bf16.p,(std::string(matched)+"/refiner-1.bf16").c_str(),rows,5376):Py_NewRef(state.p));
  Owned flat_prior(prior.p?PyObject_CallMethod(prior.p,"view","ii",rows,5376):nullptr);
  Owned operands(flat_prior.p?PyTuple_Pack(1,flat_prior.p):nullptr);
  if(!operands.p)return -1;
  Owned result(PyObject_CallObject(norm.p,operands.p));
  if(!result.p||!save(result.p,u8.p,std::string(out)+"/conditioning.bf16",int64_t(rows)*5376*2))return -1;
  Owned closed(PyObject_CallMethod(scope.p,"__exit__","OOO",Py_None,Py_None,Py_None));
  if(!closed.p)return -1;
  std::printf("original_conditioning_complete rows=%d literal_encoder=true original_projection=true original_refiners=true matched_native_frontiers=%s\n",rows,matched?"true":"false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=5&&argc!=6)return 2;
  char *end=nullptr;const long rows=std::strtol(argv[4],&end,10);
  if(end==argv[4]||*end||rows<1||rows>1024)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],static_cast<int>(rows),argc==6?argv[5]:nullptr);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
