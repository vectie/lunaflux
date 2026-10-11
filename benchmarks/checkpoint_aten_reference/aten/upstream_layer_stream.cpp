// Offline original TP1 prefill, with one complete original Block resident.
// Calls original Block.forward one literal token at a time, preserving its KV
// state. Reordering independent layers' weight residency changes no arithmetic.
// No replacement kernel, model plugin or Python dependency enters production.
#include "original_tensor_io.h"
#include "original_block_boundaries.h"
#include <cstdlib>

static bool copy_bytes(PyObject *target,PyObject *source,PyObject *u8) {
  Owned a(PyObject_CallMethod(target,"view","O",u8));
  Owned b(PyObject_CallMethod(source,"view","O",u8));
  Owned copied(a.p&&b.p?PyObject_CallMethod(a.p,"copy_","O",b.p):nullptr);
  return copied.p!=nullptr;
}

// Original convert.py publishes wo_a as BF16 after blockwise FP8 dequant.
static PyObject *output_a(PyObject *tensor,PyObject *scale) {
  Owned wide(PyObject_CallMethod(tensor,"float",nullptr));
  Owned swide(PyObject_CallMethod(scale,"float",nullptr));
  Owned w(wide.p?PyObject_CallMethod(wide.p,"view","iiii",64,128,32,128):nullptr);
  Owned s(swide.p?PyObject_CallMethod(swide.p,"view","iiii",64,1,32,1):nullptr);
  Owned product(w.p&&s.p?PyObject_CallMethod(w.p,"mul","O",s.p):nullptr);
  Owned flat(product.p?PyObject_CallMethod(product.p,"view","ii",8192,4096):nullptr);
  return flat.p?PyObject_CallMethod(flat.p,"bfloat16",nullptr):nullptr;
}

static bool load_block(PyObject *block,PyObject *open,PyObject *map,const char *root,
                       int layer,PyObject *u8) {
  Owned state(PyObject_CallMethod(block,"state_dict",nullptr));
  if(!state.p)return false;
  PyObject *key,*target;Py_ssize_t at=0;
  while(PyDict_Next(state.p,&at,&key,&target)) {
    const char *name=PyUnicode_AsUTF8(key);if(!name)return false;
    const std::string local(name),full="layers."+std::to_string(layer)+"."+local;
    Owned source(weight(open,map,root,full.c_str()));
    if(!source.p)return false;
    if(local=="attn.wo_a.weight") {
      Owned scale(weight(open,map,root,("layers."+std::to_string(layer)+".attn.wo_a.scale").c_str()));
      Owned converted(scale.p?output_a(source.p,scale.p):nullptr);
      Owned copied(converted.p?PyObject_CallMethod(target,"copy_","O",converted.p):nullptr);
      if(!copied.p)return false;
    } else if(local.ends_with(".scale")||(local.ends_with(".weight")&&
              (local.find("experts.")!=std::string::npos||local=="attn.wq_a.weight"||
               local=="attn.wq_b.weight"||local=="attn.wkv.weight"||local=="attn.wo_b.weight"||
               local=="attn.indexer.wq_b.weight"))) {
      // Shell FP4/FP8 dtypes preserve their literal payload; no implicit cast.
      if(!copy_bytes(target,source.p,u8))return false;
    } else {
      Owned copied(PyObject_CallMethod(target,"copy_","O",source.p));
      if(!copied.p)return false;
    }
  }
  // Module.to() can replace Parameter objects. Reattach the original scale
  // alias after loading, exactly as Linear.forward expects it.
  Owned modules(PyObject_CallMethod(block,"modules",nullptr));
  Owned iterator(modules.p?PyObject_GetIter(modules.p):nullptr);
  if(!iterator.p)return false;
  while(auto *raw=PyIter_Next(iterator.p)) {
    Owned module(raw);
    const int has=PyObject_HasAttrString(module.p,"scale");
    if(has<0)return false;
    if(has) {
      Owned scale(PyObject_GetAttrString(module.p,"scale"));
      if(scale.p&&scale.p!=Py_None) {
        Owned parameter(PyObject_GetAttrString(module.p,"weight"));
        if(!parameter.p||PyObject_SetAttrString(parameter.p,"scale",scale.p)<0)return false;
      }
    }
  }
  return !PyErr_Occurred();
}

static int execute(const char *inference,const char *root,const char *config_path,
                   const char *ids_path,const char *out,int layers,bool prefill,const char *ffn_input) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned fp8(PyObject_GetAttrString(torch.p,"float8_e4m3fn")),e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu"));
  Owned i64(PyObject_GetAttrString(torch.p,"int64")),fmt(PyUnicode_FromString("ue8m0"));
  Owned dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  Owned device(PyObject_CallMethod(torch.p,"set_default_device","s","cuda"));
  if(!grad.p||!threads.p||!dtype.p||!device.p||!u8.p||!i64.p||!fp8.p||!e8.p||
     PyObject_SetAttrString(model.p,"default_dtype",fp8.p)<0||
     PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0||
     PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0)return -1;
  Owned options(json_file(config_path)),cls(PyObject_GetAttrString(model.p,"ModelArgs"));
  Owned empty(PyTuple_New(0));
  Owned args(options.p&&cls.p?PyObject_Call(cls.p,empty.p,options.p):nullptr);
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  if(!args.p||!map||!open.p)return -1;
  std::ifstream file(ids_path,std::ios::binary);std::vector<int32_t> tokens;
  int32_t token;while(file.read(reinterpret_cast<char*>(&token),4))tokens.push_back(token);
  if(!file.eof()||tokens.size()!=144){PyErr_SetString(PyExc_ValueError,"144 original IDs required");return -1;}
  Owned list(PyList_New(tokens.size()));
  for(size_t i=0;i<tokens.size();++i)PyList_SET_ITEM(list.p,i,PyLong_FromLong(tokens[i]));
  Owned factory(PyObject_GetAttrString(torch.p,"tensor"));
  Owned idargs(PyTuple_Pack(1,list.p)),idkw(Py_BuildValue("{s:O,s:s}","dtype",i64.p,"device","cpu"));
  Owned flat(factory.p?PyObject_Call(factory.p,idargs.p,idkw.p):nullptr);
  Owned idview(flat.p?PyObject_CallMethod(flat.p,"view","ii",1,144):nullptr);
  Owned ids(idview.p?PyObject_CallMethod(idview.p,"to","s","cuda"):nullptr);
  Owned embedding(weight(open.p,map,root,"embed.weight",flat.p));
  Owned viewed(embedding.p?PyObject_CallMethod(embedding.p,"view","iiii",1,144,1,4096):nullptr);
  Owned h(viewed.p?PyObject_CallMethod(viewed.p,"repeat","iiii",1,1,4,1):nullptr);
  if(!h.p||!ids.p||!save(h.p,u8.p,std::string(out)+"/hidden-0.bf16",144LL*4*4096*2))return -1;
  Owned block_cls(PyObject_GetAttrString(model.p,"Block"));
  for(int layer=0;layer<layers;++layer) {
    // Scope retires this original block, its parameters and KV before the next.
    Owned ctor(Py_BuildValue("iO",layer,args.p));
    Owned block(block_cls.p?PyObject_CallObject(block_cls.p,ctor.p):nullptr);
    if(!block.p||!load_block(block.p,open.p,map,root,layer,u8.p))return -1;
    if(ffn_input) {
      if(layer!=0||layers!=1||prefill){PyErr_SetString(PyExc_ValueError,"literal layer-0 one-row FFN replay required");return -1;}
      Owned literal(literal_bf16(torch.p,bf16.p,ffn_input,144,4096));
      Owned input(literal.p?PyObject_CallMethod(literal.p,"view","iii",1,144,4096):nullptr);
      Owned ffn(PyObject_GetAttrString(block.p,"ffn"));
      Owned output(input.p?PyObject_CallMethod(torch.p,"empty_like","O",input.p):nullptr);
      if(!input.p||!ffn.p||!output.p)return -1;
      for(int position=0;position<144;++position) {
        Owned row(PyObject_CallMethod(input.p,"narrow","iii",1,position,1));
        Owned id(PyObject_CallMethod(ids.p,"narrow","iii",1,position,1));
        Owned operands(row.p&&id.p?PyTuple_Pack(2,row.p,id.p):nullptr);
        Owned result(operands.p?PyObject_CallObject(ffn.p,operands.p):nullptr);
        Owned destination(PyObject_CallMethod(output.p,"narrow","iii",1,position,1));
        Owned copied(result.p&&destination.p?PyObject_CallMethod(destination.p,"copy_","O",result.p):nullptr);
        if(!copied.p)return -1;
      }
      if(!save(output.p,u8.p,std::string(out)+"/matched-ffn-output.bf16",144LL*4096*2))return -1;
      std::puts("original_ffn_replay_complete layer=0 rows=144 submission_rows=1 original_inputs=true original_routes=true");
      return 0;
    }
    const bool observe=layer==0&&std::getenv("LUNA_ORIGINAL_BOUNDARIES");
    if(observe&&prefill){PyErr_SetString(PyExc_ValueError,"matched observer requires one-row submissions");return -1;}
    OriginalBoundaries observer;
    Owned an(observe?PyObject_GetAttrString(block.p,"attn_norm"):nullptr);
    Owned attention(observe?PyObject_GetAttrString(block.p,"attn"):nullptr);
    Owned fn(observe?PyObject_GetAttrString(block.p,"ffn_norm"):nullptr);
    Owned ffn(observe?PyObject_GetAttrString(block.p,"ffn"):nullptr);
    if(observe) {
      if(!an.p||!attention.p||!fn.p||!ffn.p)return -1;
      observer.attention_norm=an.p;observer.attention=attention.p;
      observer.ffn_norm=fn.p;observer.ffn=ffn.p;observer.block=block.p;observer.u8=u8.p;
      boundary_observer=&observer;
      PyEval_SetTrace(trace_original_boundary,Py_None);
    }
    Owned output(PyObject_CallMethod(torch.p,"empty_like","O",h.p));
    if(!output.p)return -1;
    const int step=prefill?144:1;
    for(int position=0;position<144;position+=step) {
      observer.post_count=0;
      Owned x(PyObject_CallMethod(h.p,"narrow","iii",1,position,step));
      Owned id(PyObject_CallMethod(ids.p,"narrow","iii",1,position,step));
      Owned callargs(x.p&&id.p?Py_BuildValue("OiO",x.p,position,id.p):nullptr);
      Owned y(callargs.p?PyObject_CallObject(block.p,callargs.p):nullptr);
      Owned destination(PyObject_CallMethod(output.p,"narrow","iii",1,position,step));
      Owned copied(y.p&&destination.p?PyObject_CallMethod(destination.p,"copy_","O",y.p):nullptr);
      if(!copied.p)return -1;
    }
    if(observe) {
      PyEval_SetTrace(nullptr,nullptr);boundary_observer=nullptr;
      if(!write_original_boundaries(observer,out))return -1;
      std::puts("original_layer_boundaries layer=0 rows=144 submission_rows=1 arithmetic=unchanged");
    }
    if(!save(output.p,u8.p,std::string(out)+"/hidden-"+std::to_string(layer+1)+".bf16",144LL*4*4096*2))return -1;
    Owned last(PyObject_CallMethod(output.p,"narrow","iii",1,143,1));
    Owned wide(last.p?PyObject_CallMethod(last.p,"float",nullptr):nullptr);
    Owned mean(wide.p?PyObject_CallMethod(wide.p,"mean","i",2):nullptr);
    Owned rounded(mean.p?PyObject_CallMethod(mean.p,"bfloat16",nullptr):nullptr);
    if(!rounded.p||!save(rounded.p,u8.p,std::string(out)+"/mean-layer-"+std::to_string(layer)+".bf16",8192))return -1;
    Py_SETREF(h.p,output.p);output.p=nullptr;
    std::printf("original_layer_stream layer=%d rows=144 submission_rows=%d topology=tp1 arithmetic=original_block\n",layer,step);std::fflush(stdout);
  }
  std::printf("original_layer_stream_complete layers=%d scope=original_base_prefill_without_head\n",layers);
  return 0;
}
int main(int argc,char **argv) {
  if(argc!=7&&argc!=8&&argc!=9)return 2;
  if(argc==8&&std::string(argv[7])!="--prefill")return 2;
  if(argc==9&&std::string(argv[7])!="--ffn-input")return 2;
  const int layers=std::atoi(argv[6]);if(layers<1||layers>43)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],argv[5],layers,argc==8,argc==9?argv[8]:nullptr);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
