// Offline matched first-layer component replay. Calls the installed original
// Block.hc_pre/RMSNorm and kernel.hc_split_sinkhorn, not reconstructed math.
#include "original_tensor_io.h"

static int execute(const char *inference,const char *root,const char *input,const char *out) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors")),types(PyImport_ImportModule("types"));
  if(!torch.p||!model.p||!safe.p||!types.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned u8(PyObject_GetAttrString(torch.p,"uint8")),i64(PyObject_GetAttrString(torch.p,"int64"));
  Owned f32(PyObject_GetAttrString(torch.p,"float32")),bf16(PyObject_GetAttrString(torch.p,"bfloat16"));
  if(!grad.p||!threads.p||!u8.p||!i64.p||!f32.p||!bf16.p)return -1;
  std::ifstream file(input,std::ios::binary);
  std::vector<int32_t> ids;int32_t id;
  while(file.read(reinterpret_cast<char *>(&id),4))ids.push_back(id);
  if(!file.eof()||ids.empty()||ids.size()>144){PyErr_SetString(PyExc_ValueError,"literal prompt IDs required");return -1;}
  Owned list(PyList_New(ids.size()));
  for(size_t i=0;i<ids.size();++i)PyList_SET_ITEM(list.p,i,PyLong_FromLong(ids[i]));
  Owned factory(PyObject_GetAttrString(torch.p,"tensor"));
  Owned args(PyTuple_Pack(1,list.p)),kw(Py_BuildValue("{s:O}","dtype",i64.p));
  Owned indices(PyObject_Call(factory.p,args.p,kw.p));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  if(!indices.p||!map||!open.p)return -1;
  Owned embed(weight(open.p,map,root,"embed.weight",indices.p));
  const int rows=static_cast<int>(ids.size());
  Owned view(embed.p?PyObject_CallMethod(embed.p,"view","iiii",1,rows,1,4096):nullptr);
  Owned expanded(view.p?PyObject_CallMethod(view.p,"expand","iiii",1,rows,4,4096):nullptr);
  Owned streams(expanded.p?PyObject_CallMethod(expanded.p,"contiguous",nullptr):nullptr);
  Owned fn(weight(open.p,map,root,"layers.0.hc_attn_fn"));
  Owned scale(weight(open.p,map,root,"layers.0.hc_attn_scale"));
  Owned base(weight(open.p,map,root,"layers.0.hc_attn_base"));
  Owned norm(weight(open.p,map,root,"layers.0.attn_norm.weight"));
  if(!streams.p||!fn.p||!scale.p||!base.p||!norm.p)return -1;
  Owned ns(PyObject_GetAttrString(types.p,"SimpleNamespace")),empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:d,s:i,s:i,s:d}","norm_eps",1e-6,"hc_mult",4,"hc_sinkhorn_iters",20,"hc_eps",1e-6));
  Owned self(PyObject_Call(ns.p,empty.p,options.p));
  Owned cls(PyObject_GetAttrString(model.p,"Block"));
  Owned method(cls.p?PyObject_GetAttrString(cls.p,"hc_pre"):nullptr);
  Owned operands(PyTuple_Pack(5,self.p,streams.p,fn.p,scale.p,base.p));
  Owned result(method.p?PyObject_CallObject(method.p,operands.p):nullptr);
  if(!result.p||!PyTuple_Check(result.p)||PyTuple_Size(result.p)!=3)return -1;
  auto *collapsed=PyTuple_GetItem(result.p,0),*post=PyTuple_GetItem(result.p,1),*comb=PyTuple_GetItem(result.p,2);
  Owned norm_cls(PyObject_GetAttrString(model.p,"RMSNorm"));
  Owned rms(norm_cls.p?PyObject_CallFunction(norm_cls.p,"id",4096,1e-6):nullptr);
  Owned gpu(rms.p?PyObject_CallMethod(rms.p,"to","s","cuda"):nullptr);
  Owned parameter(gpu.p?PyObject_GetAttrString(gpu.p,"weight"):nullptr);
  Owned wide_norm(norm.p?PyObject_CallMethod(norm.p,"float",nullptr):nullptr);
  Owned copied(parameter.p&&wide_norm.p?PyObject_CallMethod(parameter.p,"copy_","O",wide_norm.p):nullptr);
  Owned rms_args(PyTuple_Pack(1,collapsed));
  Owned normalized(copied.p?PyObject_CallObject(gpu.p,rms_args.p):nullptr);
  Owned cuda(PyObject_GetAttrString(torch.p,"cuda"));
  Owned sync(cuda.p?PyObject_CallMethod(cuda.p,"synchronize",nullptr):nullptr);
  if(!normalized.p||!sync.p)return -1;
  if(!save(streams.p,u8.p,std::string(out)+"/streams.bf16",int64_t(rows)*4*4096*2)||
     !save(fn.p,u8.p,std::string(out)+"/function.f32",24*16384*4)||
     !save(scale.p,u8.p,std::string(out)+"/scale.f32",3*4)||
     !save(base.p,u8.p,std::string(out)+"/base.f32",24*4)||
     !save(norm.p,u8.p,std::string(out)+"/norm.bf16",4096*2)||
     !save(collapsed,u8.p,std::string(out)+"/collapsed.bf16",int64_t(rows)*4096*2)||
     !save(post,u8.p,std::string(out)+"/post.f32",int64_t(rows)*4*4)||
     !save(comb,u8.p,std::string(out)+"/combination.f32",int64_t(rows)*16*4)||
     !save(normalized.p,u8.p,std::string(out)+"/normalized.bf16",int64_t(rows)*4096*2))return -1;
  std::printf("original_hyper_complete rows=%d first_layer=0 operands=original_literal arithmetic=Block.hc_pre+RMSNorm full_model=false\n",rows);
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=5)return 2;
  Py_Initialize();
  const int result=execute(argv[1],argv[2],argv[3],argv[4]);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
