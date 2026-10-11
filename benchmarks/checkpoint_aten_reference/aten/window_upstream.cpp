// Offline original rotary/cache/window prefix on literal query/KV operands.
// Calls installed original functions; no replacement attention mathematics.
#include "original_tensor_io.h"
#include <cmath>

static int execute(const char *inference,const char *root,const char *input,const char *output,int rows) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned kernel(PyImport_ImportModule("kernel")),safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!kernel.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  Owned dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  Owned e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu")),fmt(PyUnicode_FromString("ue8m0"));
  if(!grad.p||!threads.p||!dtype.p||!u8.p||!e8.p||!fmt.p)return -1;
  Owned qflat(literal_bf16(torch.p,bf16.p,(std::string(input)+"/q-head.bf16").c_str(),rows,32768));
  Owned q(qflat.p?PyObject_CallMethod(qflat.p,"view","iiii",1,rows,64,512):nullptr);
  Owned kv(literal_bf16(torch.p,bf16.p,(std::string(input)+"/kv-norm.bf16").c_str(),rows,512));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned sink(map&&open.p?weight(open.p,map,root,"layers.0.attn.attn_sink"):nullptr);
  Owned factory(PyObject_GetAttrString(model.p,"precompute_freqs_cis"));
  Owned args(Py_BuildValue("(iiidddd)",64,rows,0,10000.0,16.0,32.0,1.0));
  Owned cpu_freq(factory.p?PyObject_CallObject(factory.p,args.p):nullptr);
  Owned freq(cpu_freq.p?PyObject_CallMethod(cpu_freq.p,"to","s","cuda"):nullptr);
  Owned rotary(PyObject_GetAttrString(model.p,"apply_rotary_emb"));
  Owned quantize(PyObject_GetAttrString(kernel.p,"act_quant"));
  Owned window(PyObject_GetAttrString(model.p,"get_window_topk_idxs"));
  Owned attention(PyObject_GetAttrString(kernel.p,"sparse_attn"));
  if(!q.p||!kv.p||!sink.p||!freq.p||!rotary.p||!quantize.p||!window.p||!attention.p)return -1;
  if(!save(q.p,u8.p,std::string(output)+"/q-input.bf16",int64_t(rows)*32768*2)||
    !save(kv.p,u8.p,std::string(output)+"/kv-input.bf16",int64_t(rows)*512*2)||
    !save(sink.p,u8.p,std::string(output)+"/sink.f32",64*4)||
    !save(freq.p,u8.p,std::string(output)+"/frequencies.complex64",int64_t(rows)*32*8))return -1;
  Owned qs(PyObject_CallMethod(q.p,"narrow","iii",-1,448,64));
  Owned ks(PyObject_CallMethod(kv.p,"narrow","iii",-1,448,64));
  Owned qa(qs.p?PyTuple_Pack(2,qs.p,freq.p):nullptr),ka(ks.p?PyTuple_Pack(2,ks.p,freq.p):nullptr);
  Owned qr(qa.p?PyObject_CallObject(rotary.p,qa.p):nullptr),kr(ka.p?PyObject_CallObject(rotary.p,ka.p):nullptr);
  if(!qr.p||!kr.p||!save(q.p,u8.p,std::string(output)+"/q-rope.bf16",int64_t(rows)*32768*2)||
    !save(kv.p,u8.p,std::string(output)+"/kv-rope.bf16",int64_t(rows)*512*2))return -1;
  Owned prefix(PyObject_CallMethod(kv.p,"narrow","iii",-1,0,448));
  Owned quant_args(prefix.p?Py_BuildValue("(OiOOO)",prefix.p,64,fmt.p,e8.p,Py_True):nullptr);
  Owned simulated(quant_args.p?PyObject_CallObject(quantize.p,quant_args.p):nullptr);
  Owned win_args(Py_BuildValue("(iiii)",128,1,rows,0));
  Owned indices(PyObject_CallObject(window.p,win_args.p));
  Owned idx_gpu(indices.p?PyObject_CallMethod(indices.p,"to","s","cuda"):nullptr);
  if(!simulated.p||!idx_gpu.p||!save(kv.p,u8.p,std::string(output)+"/kv-sim.bf16",int64_t(rows)*512*2)||
    !save(idx_gpu.p,u8.p,std::string(output)+"/indices.i32",int64_t(rows)*128*4))return -1;
  Owned attn_args(Py_BuildValue("(OOOOd)",q.p,kv.p,sink.p,idx_gpu.p,1.0/std::sqrt(512.0)));
  Owned result(PyObject_CallObject(attention.p,attn_args.p));
  if(!result.p||!save(result.p,u8.p,std::string(output)+"/attention.bf16",int64_t(rows)*32768*2))return -1;
  Owned suffix(PyObject_CallMethod(result.p,"narrow","iii",-1,448,64));
  Owned inv_args(suffix.p?PyTuple_Pack(3,suffix.p,freq.p,Py_True):nullptr);
  Owned inverse(inv_args.p?PyObject_CallObject(rotary.p,inv_args.p):nullptr);
  if(!inverse.p||!save(result.p,u8.p,std::string(output)+"/inverse.bf16",int64_t(rows)*32768*2))return -1;
  std::printf("original_window_complete rows=%d heads=64 window=128 probability=bf16-kv64 original_functions=true tp=1 full_model=false\n",rows);
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=6)return 2;
  char *end=nullptr;long rows=std::strtol(argv[5],&end,10);
  // This replay fixes a complete 128-slot original prefill read set.
  if(end==argv[5]||!end||*end||rows<128||rows>144)return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],static_cast<int>(rows));
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
