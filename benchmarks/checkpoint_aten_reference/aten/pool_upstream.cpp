// Offline original learned-compressor replay on a literal original row corpus.
// Forward hooks observe pooled and normalized tensors without replacing math.
#include "original_tensor_io.h"

struct Capture { const char *out; PyObject *u8; int rows; };
static PyObject *observe_pool(PyObject *self,PyObject *args) {
  auto *capture=static_cast<Capture *>(PyCapsule_GetPointer(self,"pool.capture"));
  if(!capture)return nullptr;
  const bool before=PyTuple_Size(args)==2;
  PyObject *tensor=nullptr;
  if(before) {
    auto *input=PyTuple_GetItem(args,1);
    if(!PyTuple_Check(input)||PyTuple_Size(input)!=1)return nullptr;
    tensor=PyTuple_GetItem(input,0);
  } else if(PyTuple_Size(args)==3) tensor=PyTuple_GetItem(args,2);
  if(!tensor||!save(tensor,capture->u8,std::string(capture->out)+
      (before?"/pooled.bf16":"/normalized.bf16"),int64_t(capture->rows)*512*2))return nullptr;
  Py_RETURN_NONE;
}
static PyMethodDef observer={"observe_pool",observe_pool,METH_VARARGS,"Observe original boundaries."};

static int execute(const char *inference,const char *root,const char *input,const char *out,int rows,int ratio) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p));
  Owned e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu")),fmt(PyUnicode_FromString("ue8m0"));
  if(!grad.p||!threads.p||!dtype.p||!e8.p||!fmt.p||
     PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0||PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0)return -1;
  Owned x(literal_bf16(torch.p,bf16.p,input,rows,4096));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned args_cls(PyObject_GetAttrString(model.p,"ModelArgs")),empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i,s:i,s:i,s:d}","dim",4096,"rope_head_dim",64,
    "max_batch_size",1,"max_seq_len",rows,"norm_eps",1e-6));
  Owned config(PyObject_Call(args_cls.p,empty.p,options.p));
  Owned cls(PyObject_GetAttrString(model.p,"Compressor"));
  Owned ctor(Py_BuildValue("Oiii",config.p,ratio,512,0));
  Owned owner(cls.p?PyObject_CallObject(cls.p,ctor.p):nullptr);
  Owned gpu(owner.p?PyObject_CallMethod(owner.p,"to","s","cuda"):nullptr);
  if(!x.p||!map||!open.p||!gpu.p)return -1;
  const int layer=ratio==4?2:3,channels=ratio==4?1024:512;
  const std::string prefix="layers."+std::to_string(layer)+".attn.compressor.";
  Owned xf(PyObject_CallMethod(x.p,"float",nullptr));
  for(const char *name:{"wkv","wgate","norm","ape"}) {
    const bool position=std::string(name)=="ape";
    const std::string key=prefix+name+(position?"":".weight");
    Owned payload(weight(open.p,map,root,key.c_str()));
    Owned module(PyObject_GetAttrString(gpu.p,name));
    Owned target(position?Py_XNewRef(module.p):(module.p?PyObject_GetAttrString(module.p,"weight"):nullptr));
    Owned copied(target.p&&payload.p?PyObject_CallMethod(target.p,"copy_","O",payload.p):nullptr);
    const int64_t bytes=position?int64_t(ratio)*channels*4:
      std::string(name)=="norm"?512*2:int64_t(channels)*4096*2;
    if(!copied.p||!save(payload.p,u8.p,std::string(out)+"/"+name+(position?".f32":".bf16"),bytes))return -1;
    if(std::string(name)=="wkv"||std::string(name)=="wgate") {
      Owned operand(PyTuple_Pack(1,xf.p));
      Owned result(PyObject_CallObject(module.p,operand.p));
      if(!result.p||!save(result.p,u8.p,std::string(out)+"/"+name+"-projection.f32",int64_t(rows)*channels*4))return -1;
    }
  }
  Owned norm(PyObject_GetAttrString(gpu.p,"norm"));
  Capture capture{out,u8.p,rows/ratio};
  Owned capsule(PyCapsule_New(&capture,"pool.capture",nullptr));
  Owned hook(PyCFunction_NewEx(&observer,capsule.p,nullptr));
  Owned pre(PyObject_CallMethod(norm.p,"register_forward_pre_hook","O",hook.p));
  Owned post(PyObject_CallMethod(norm.p,"register_forward_hook","O",hook.p));
  Owned zeros(PyObject_GetAttrString(torch.p,"zeros"));
  Owned shape(Py_BuildValue("iii",1,rows/ratio,512)),zero_args(PyTuple_Pack(1,shape.p));
  Owned zero_options(Py_BuildValue("{s:O,s:s}","dtype",bf16.p,"device","cuda"));
  Owned cache(PyObject_Call(zeros.p,zero_args.p,zero_options.p));
  Owned frequency(PyObject_GetAttrString(model.p,"precompute_freqs_cis"));
  Owned frequency_args(Py_BuildValue("iiidddd",64,rows,65536,160000.0,16.0,32.0,1.0));
  Owned freq_cpu(PyObject_CallObject(frequency.p,frequency_args.p));
  Owned freq(freq_cpu.p?PyObject_CallMethod(freq_cpu.p,"to","s","cuda"):nullptr);
  if(!pre.p||!post.p||!cache.p||!freq.p||
     PyObject_SetAttrString(gpu.p,"kv_cache",cache.p)<0||PyObject_SetAttrString(gpu.p,"freqs_cis",freq.p)<0)return -1;
  Owned operands(Py_BuildValue("Oi",x.p,0));
  Owned result(PyObject_CallObject(gpu.p,operands.p));
  if(!result.p||!save(result.p,u8.p,std::string(out)+"/transformed.bf16",int64_t(rows/ratio)*512*2))return -1;
  std::printf("original_pool_complete parameter_layer=%d ratio=%d rows=%d input=literal_original_ffn_corpus full_model=false\n",layer,ratio,rows);
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=7)return 2;
  const int rows=std::atoi(argv[5]),ratio=std::atoi(argv[6]);
  if(rows!=144||(ratio!=4&&ratio!=128))return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],rows,ratio);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
