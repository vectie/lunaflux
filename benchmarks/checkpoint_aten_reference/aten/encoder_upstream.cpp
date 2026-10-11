// Offline invocation of the installed original H3 text decoder layers.
// Only weight residency changes: one original layer is live at a time.
// No encoder arithmetic is reimplemented or linked into production.
#include <Python.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *value=nullptr) : p(value) {}
  ~Owned() { Py_XDECREF(p); }
  Owned(const Owned &)=delete;
};

static PyObject *call(PyObject *function,PyObject *args,PyObject *kwargs=nullptr) {
  return function&&args ? PyObject_Call(function,args,kwargs) : nullptr;
}
static PyObject *json_file(const std::string &path) {
  std::ifstream file(path);
  if(!file) { PyErr_SetString(PyExc_OSError,"original metadata cannot be read"); return nullptr; }
  const std::string bytes((std::istreambuf_iterator<char>(file)),{});
  Owned json(PyImport_ImportModule("json"));
  return json.p?PyObject_CallMethod(json.p,"loads","s",bytes.c_str()):nullptr;
}

// Each safe_open backing store and its CPU tensor view are released after a
// blocking copy. This avoids a full CPU state dictionary on unified RAM.
static PyObject *weight(PyObject *open,PyObject *map,const char *root,const std::string &name) {
  auto *shard=PyDict_GetItemString(map,name.c_str());
  if(!shard) { PyErr_Format(PyExc_KeyError,"original weight missing: %s",name.c_str());return nullptr; }
  const char *filename=PyUnicode_AsUTF8(shard);
  if(!filename)return nullptr;
  Owned path(PyUnicode_FromString((std::string(root)+"/"+filename).c_str()));
  Owned args(PyTuple_Pack(1,path.p));
  Owned kwargs(Py_BuildValue("{s:s,s:s}","framework","pt","device","cpu"));
  Owned reader(call(open,args.p,kwargs.p));
  Owned tensor(reader.p?PyObject_CallMethod(reader.p,"get_tensor","s",name.c_str()):nullptr);
  // Clone only the named tensor: callers immediately copy/release it. A CPU
  // clone owns its bytes independently of safetensors' mmap lifetime.
  return tensor.p?PyObject_CallMethod(tensor.p,"clone",nullptr):nullptr;
}

static bool load_parameter(PyObject *layer,PyObject *open,PyObject *map,const char *root,
                           const char *target,const std::string &source,const char *part=nullptr) {
  Owned parameter(PyObject_GetAttrString(layer,target));
  if(!parameter.p)return false;
  Owned tensor(weight(open,map,root,source));
  if(!tensor.p)return false;
  Owned loader(part?PyObject_GetAttrString(parameter.p,"weight_loader"):nullptr);
  Owned args(part?Py_BuildValue("OOs",parameter.p,tensor.p,part):nullptr);
  Owned copied(part?call(loader.p,args.p):PyObject_CallMethod(parameter.p,"copy_","O",tensor.p));
  return copied.p!=nullptr;
}

static bool load_layer(PyObject *layer,PyObject *open,PyObject *map,const char *root,int ordinal) {
  const std::string prefix="model.language_model.layers."+std::to_string(ordinal)+".";
  Owned attention(PyObject_GetAttrString(layer,"self_attn"));
  Owned mlp(PyObject_GetAttrString(layer,"mlp"));
  Owned qkv(attention.p?PyObject_GetAttrString(attention.p,"qkv_proj"):nullptr);
  Owned gate_up(mlp.p?PyObject_GetAttrString(mlp.p,"gate_up_proj"):nullptr);
  if(!attention.p||!mlp.p||!qkv.p||!gate_up.p)return false;
  for(const char *name : {"input_layernorm","post_attention_layernorm"}) {
    Owned norm(PyObject_GetAttrString(layer,name));
    if(!norm.p||!load_parameter(norm.p,open,map,root,"weight",prefix+name+".weight"))return false;
  }
  for(const char *name : {"q_norm","k_norm","o_proj"}) {
    Owned child(PyObject_GetAttrString(attention.p,name));
    if(!child.p||!load_parameter(child.p,open,map,root,"weight",prefix+"self_attn."+name+".weight"))return false;
  }
  for(const char *part : {"q","k","v"}) {
    if(!load_parameter(qkv.p,open,map,root,"weight",prefix+"self_attn."+part+"_proj.weight",part))return false;
  }
  // Original merged MLP loader's shard IDs are integers (0=gate, 1=up).
  Owned parameter(PyObject_GetAttrString(gate_up.p,"weight"));
  Owned loader(parameter.p?PyObject_GetAttrString(parameter.p,"weight_loader"):nullptr);
  if(!loader.p)return false;
  for(int shard=0;shard<2;++shard) {
    Owned tensor(weight(open,map,root,prefix+"mlp."+(shard?"up":"gate")+"_proj.weight"));
    Owned args(tensor.p?Py_BuildValue("OOi",parameter.p,tensor.p,shard):nullptr);
    Owned copied(call(loader.p,args.p));
    if(!copied.p)return false;
  }
  Owned down(PyObject_GetAttrString(mlp.p,"down_proj"));
  return down.p&&load_parameter(down.p,open,map,root,"weight",prefix+"mlp.down_proj.weight");
}

static bool write_tensor(PyObject *tensor,PyObject *byte_dtype,const std::string &path,int64_t expected) {
  Owned cpu(PyObject_CallMethod(tensor,"cpu",nullptr));
  Owned contiguous(cpu.p?PyObject_CallMethod(cpu.p,"contiguous",nullptr):nullptr);
  Owned raw(contiguous.p?PyObject_CallMethod(contiguous.p,"view","O",byte_dtype):nullptr);
  Owned array(raw.p?PyObject_CallMethod(raw.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p)return false;
  if(PyBytes_Size(bytes.p)!=expected) { PyErr_SetString(PyExc_ValueError,"original hidden extent differs");return false; }
  auto *file=std::fopen(path.c_str(),"wbx");
  if(!file) { PyErr_SetFromErrnoWithFilename(PyExc_OSError,path.c_str());return false; }
  const bool written=std::fwrite(PyBytes_AsString(bytes.p),1,expected,file)==static_cast<size_t>(expected);
  const bool closed=std::fclose(file)==0;
  if(!written||!closed) { PyErr_SetString(PyExc_OSError,"original hidden write failed");return false; }
  return true;
}

static int execute(const char *root,const char *input,const char *output,int layers,
                   int replay_layer=-1,const char *replay_input=nullptr) {
  Owned torch(PyImport_ImportModule("torch"));
  Owned module(PyImport_ImportModule("vllm_omni.diffusion.models.minimax_h3.encoder"));
  Owned transformers(PyImport_ImportModule("transformers"));
  Owned safe(PyImport_ImportModule("safetensors"));
  Owned types(PyImport_ImportModule("types"));
  if(!torch.p||!module.p||!transformers.p||!safe.p||!types.p)return -1;
  Owned no_grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16"));
  Owned u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned i64(PyObject_GetAttrString(torch.p,"int64"));
  Owned dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  if(!no_grad.p||!threads.p||!dtype.p||!u8.p||!i64.p)return -1;
  Owned backends(PyObject_GetAttrString(torch.p,"backends"));
  Owned cuda(backends.p?PyObject_GetAttrString(backends.p,"cuda"):nullptr);
  Owned cudnn(cuda.p?PyObject_CallMethod(cuda.p,"enable_cudnn_sdp","O",Py_True):nullptr);
  Owned auto_config(PyObject_GetAttrString(transformers.p,"AutoConfig"));
  Owned config(auto_config.p?PyObject_CallMethod(auto_config.p,"from_pretrained","s",root):nullptr);
  Owned text(config.p?PyObject_GetAttrString(config.p,"text_config"):nullptr);
  Owned cls(PyObject_GetAttrString(types.p,"SimpleNamespace"));
  Owned empty(PyTuple_New(0));
  Owned options(Py_BuildValue("{s:i,s:i}","rank_in_group",0,"world_size",1));
  Owned group(call(cls.p,empty.p,options.p));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  if(!cudnn.p||!text.p||!group.p||!map||!open.p)return -1;
  std::ifstream ids_file(input,std::ios::binary);
  std::vector<int32_t> ids;
  int32_t id;
  while(ids_file.read(reinterpret_cast<char *>(&id),sizeof(id)))ids.push_back(id);
  if(!ids_file.eof()||ids.empty()||ids.size()>1024) { PyErr_SetString(PyExc_ValueError,"bounded literal IDs required");return -1; }
  Owned list(PyList_New(ids.size()));
  for(size_t i=0;i<ids.size();++i)PyList_SET_ITEM(list.p,i,PyLong_FromLong(ids[i]));
  Owned tensor_fn(PyObject_GetAttrString(torch.p,"tensor"));
  Owned id_args(PyTuple_Pack(1,list.p));
  Owned id_options(Py_BuildValue("{s:O}","dtype",i64.p));
  Owned token_ids(call(tensor_fn.p,id_args.p,id_options.p));
  const int rows=static_cast<int>(ids.size());
  const int64_t bytes=static_cast<int64_t>(rows)*5120*2;
  Owned hidden;
  if(replay_layer>=0) {
    // Literal BF16 input, not re-encoding its prefix or substituting F32 values.
    // The bytearray owner remains live until the blocking CUDA copy retires.
    std::ifstream file(replay_input,std::ios::binary);
    const std::vector<char> data((std::istreambuf_iterator<char>(file)),{});
    if(!file||static_cast<int64_t>(data.size())!=bytes) {
      PyErr_SetString(PyExc_ValueError,"matched layer input extent differs");return -1;
    }
    Owned buffer(PyByteArray_FromStringAndSize(data.data(),data.size()));
    Owned frombuffer(PyObject_GetAttrString(torch.p,"frombuffer"));
    Owned buffer_args(buffer.p?PyTuple_Pack(1,buffer.p):nullptr);
    Owned buffer_options(Py_BuildValue("{s:O}","dtype",bf16.p));
    Owned flat(call(frombuffer.p,buffer_args.p,buffer_options.p));
    Owned batch(flat.p?PyObject_CallMethod(flat.p,"view","iii",1,rows,5120):nullptr);
    hidden.p=batch.p?PyObject_CallMethod(batch.p,"to","s","cuda"):nullptr;
  } else {
    Owned embedding(weight(open.p,map,root,"model.language_model.embed_tokens.weight"));
    Owned selected(embedding.p&&token_ids.p?PyObject_CallMethod(embedding.p,"index_select","iO",0,token_ids.p):nullptr);
    Owned batch(selected.p?PyObject_CallMethod(selected.p,"unsqueeze","i",0):nullptr);
    hidden.p=batch.p?PyObject_CallMethod(batch.p,"to","s","cuda"):nullptr;
  }
  if(!hidden.p)return -1;
  Owned positions(PyObject_CallMethod(torch.p,"arange","i",rows));
  Owned position_view(positions.p?PyObject_CallMethod(positions.p,"view","iii",1,1,rows):nullptr);
  Owned expanded(position_view.p?PyObject_CallMethod(position_view.p,"expand","iii",3,1,rows):nullptr);
  Owned gpu_positions(expanded.p?PyObject_CallMethod(expanded.p,"to","s","cuda"):nullptr);
  Owned rotary_cls(PyObject_GetAttrString(module.p,"MiniMaxH3Qwen3VLTextRotaryEmbedding"));
  Owned rotary_args(PyTuple_Pack(1,text.p));
  Owned rotary(call(rotary_cls.p,rotary_args.p));
  Owned position_args(hidden.p&&gpu_positions.p?PyTuple_Pack(2,hidden.p,gpu_positions.p):nullptr);
  Owned tables(call(rotary.p,position_args.p));
  Owned layer_cls(PyObject_GetAttrString(module.p,"MiniMaxH3Qwen3VLTextDecoderLayer"));
  Owned layer_args(PyTuple_Pack(3,group.p,text.p,bf16.p));
  if(!tables.p||!layer_cls.p||!layer_args.p)return -1;
  const int first=replay_layer>=0?replay_layer:0;
  if(!write_tensor(hidden.p,u8.p,std::string(output)+"/hidden-"+std::to_string(first)+".bf16",bytes))return -1;
  for(int layer=first;layer<first+layers;++layer) {
    Owned cpu_layer(call(layer_cls.p,layer_args.p));
    Owned gpu(cpu_layer.p?PyObject_CallMethod(cpu_layer.p,"to","s","cuda"):nullptr);
    if(!gpu.p||!load_layer(gpu.p,open.p,map,root,layer))return -1;
    Owned args(PyTuple_Pack(2,hidden.p,tables.p));
    Owned next(call(gpu.p,args.p));
    if(!next.p||!write_tensor(next.p,u8.p,std::string(output)+"/hidden-"+std::to_string(layer+1)+".bf16",bytes))return -1;
    Py_SETREF(hidden.p,next.p);next.p=nullptr;
    std::printf("original_encoder_layer=%d rows=%d arithmetic=installed_module\n",layer+1,rows);std::fflush(stdout);
  }
  std::printf("original_encoder_complete layers=%d rows=%d first_layer=%d matched_input=%s output_boundary=unnormalized_hidden residency=one_layer\n",
              layers,rows,first,replay_layer>=0?"true":"false");
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=5&&argc!=8) { std::fputs("ORIGINAL_TEXT_ROOT LITERAL_IDS_I32 NEW_OUTPUT_DIR LAYERS [--replay-layer ZERO_BASED_LAYER INPUT_BF16]\n",stderr);return 2; }
  char *end=nullptr;const auto count=std::strtol(argv[4],&end,10);
  if(!end||*end||count<1||count>50)return 2;
  int first=-1;
  if(argc==8) {
    if(std::string(argv[5])!="--replay-layer"||count!=1)return 2;
    const auto ordinal=std::strtol(argv[6],&end,10);
    if(end==argv[6]||!end||*end||ordinal<0||ordinal>=50)return 2;
    first=static_cast<int>(ordinal);
  }
  Py_Initialize();
  int result=execute(argv[1],argv[2],argv[3],static_cast<int>(count),first,argc==8?argv[7]:nullptr);
  if(PyErr_Occurred()) { PyErr_Print();result=-1; }
  const int finalized=Py_FinalizeEx();
  return result||finalized<0?1:0;
}
