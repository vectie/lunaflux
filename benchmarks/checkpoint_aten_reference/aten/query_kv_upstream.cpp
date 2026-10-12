// Offline original FP8 query/KV prefix on one exact original normalized input.
// The original blockwise quantizer and matrix kernels are called unchanged.
#include "original_tensor_io.h"

static PyObject *learned_norm(PyObject *model,PyObject *input,PyObject *weights,int width) {
  Owned cls(PyObject_GetAttrString(model,"RMSNorm"));
  Owned object(cls.p?PyObject_CallFunction(cls.p,"id",width,1e-6):nullptr);
  Owned gpu(object.p?PyObject_CallMethod(object.p,"to","s","cuda"):nullptr);
  Owned target(gpu.p?PyObject_GetAttrString(gpu.p,"weight"):nullptr);
  Owned wide(PyObject_CallMethod(weights,"float",nullptr));
  Owned copied(target.p&&wide.p?PyObject_CallMethod(target.p,"copy_","O",wide.p):nullptr);
  Owned args(PyTuple_Pack(1,input));
  return copied.p?PyObject_CallObject(gpu.p,args.p):nullptr;
}

static void shape(PyObject *value,const char *stage,int row) {
  if(row!=0)return;
  Owned dims(PyObject_GetAttrString(value,"shape"));
  Owned text(dims.p?PyObject_Str(dims.p):nullptr);
  if(text.p)std::printf("original_query_operand stage=%s shape=%s\n",stage,PyUnicode_AsUTF8(text.p));
}

static int execute(const char *inference,const char *root,const char *input,const char *output,int rows,int layer,bool one_row) {
  Owned path(PyUnicode_FromString(inference));
  if(!path.p||PyList_Insert(PySys_GetObject("path"),0,path.p)<0)return -1;
  Owned torch(PyImport_ImportModule("torch")),model(PyImport_ImportModule("model"));
  Owned safe(PyImport_ImportModule("safetensors"));
  if(!torch.p||!model.p||!safe.p)return -1;
  Owned grad(PyObject_CallMethod(torch.p,"set_grad_enabled","O",Py_False));
  Owned threads(PyObject_CallMethod(torch.p,"set_num_threads","i",2));
  Owned bf16(PyObject_GetAttrString(torch.p,"bfloat16")),u8(PyObject_GetAttrString(torch.p,"uint8"));
  // generate.py selects BF16 before invoking the original quantized kernels.
  // Their output allocation follows the framework default dtype.
  Owned default_dtype(bf16.p?PyObject_CallMethod(torch.p,"set_default_dtype","O",bf16.p):nullptr);
  Owned e8(PyObject_GetAttrString(torch.p,"float8_e8m0fnu")),fmt(PyUnicode_FromString("ue8m0"));
  if(!grad.p||!threads.p||!bf16.p||!default_dtype.p||!u8.p||!e8.p||!fmt.p||
    PyObject_SetAttrString(model.p,"scale_fmt",fmt.p)<0||PyObject_SetAttrString(model.p,"scale_dtype",e8.p)<0)return -1;
  Owned x(literal_bf16(torch.p,bf16.p,input,rows,4096));
  Owned index(json_file(std::string(root)+"/model.safetensors.index.json"));
  auto *map=index.p?PyDict_GetItemString(index.p,"weight_map"):nullptr;
  Owned open(PyObject_GetAttrString(safe.p,"safe_open"));
  Owned linear(PyObject_GetAttrString(model.p,"linear"));
  if(!x.p||!map||!open.p||!linear.p)return -1;
  const char *names[3]={"wq_a","wq_b","wkv"};
  const int in[3]={4096,1024,4096},out[3]={1024,32768,512};
  PyObject *weights[3]={},*scales[3]={};
  struct Matrices {
    PyObject **w,**s;
    ~Matrices(){for(int i=0;i<3;++i){Py_XDECREF(w[i]);Py_XDECREF(s[i]);}}
  } matrices{weights,scales};
  for(int i=0;i<3;++i) {
    const std::string prefix="layers."+std::to_string(layer)+".attn."+std::string(names[i]);
    weights[i]=weight(open.p,map,root,(prefix+".weight").c_str());
    scales[i]=weight(open.p,map,root,(prefix+".scale").c_str());
    if(!weights[i]||!scales[i]||PyObject_SetAttrString(weights[i],"scale",scales[i])<0)return -1;
    if(!save(weights[i],u8.p,std::string(output)+"/"+names[i]+".fp8",int64_t(in[i])*out[i])||
       !save(scales[i],u8.p,std::string(output)+"/"+names[i]+".scale",int64_t(in[i]/128)*(out[i]/128)))return -1;
  }
  const std::string prefix="layers."+std::to_string(layer)+".attn.";
  Owned qweight(weight(open.p,map,root,(prefix+"q_norm.weight").c_str()));
  Owned kweight(weight(open.p,map,root,(prefix+"kv_norm.weight").c_str()));
  if(!qweight.p||!kweight.p)return -1;
  const char *stages[6]={"q-a","q-norm","q-b","q-head","kv","kv-norm"};
  const int widths[6]={1024,1024,32768,32768,512,512};
  const int step=one_row?1:rows;
  Owned pieces[6];
  for(auto &list:pieces){list.p=PyList_New(rows/step);if(!list.p)return -1;}
  for(int row=0;row<rows;row+=step) {
  // Literal inputs retain [batch=1, sequence, hidden], like Block.forward.
  Owned part(PyObject_CallMethod(x.p,"narrow","iii",1,row,step));
  if(!part.p)return -1;
  shape(part.p,"input",row);
  Owned qa_args(part.p?PyTuple_Pack(2,part.p,weights[0]):nullptr),kv_args(part.p?PyTuple_Pack(2,part.p,weights[2]):nullptr);
  Owned qa(PyObject_CallObject(linear.p,qa_args.p));
  if(!qa.p){std::fprintf(stderr,"original query failure row=%d stage=q-a\n",row);return -1;}
  shape(qa.p,"q-a",row);
  Owned qn(qa.p?learned_norm(model.p,qa.p,qweight.p,1024):nullptr);
  if(!qn.p){std::fprintf(stderr,"original query failure row=%d stage=q-norm\n",row);return -1;}
  shape(qn.p,"q-norm",row);
  shape(qweight.p,"q-norm-weight",row);
  Owned qb_args(qn.p?PyTuple_Pack(2,qn.p,weights[1]):nullptr);
  Owned qb(qb_args.p?PyObject_CallObject(linear.p,qb_args.p):nullptr);
  if(!qb.p){std::fprintf(stderr,"original query failure row=%d stage=q-b\n",row);return -1;}
  shape(qb.p,"q-b",row);
  Owned heads(qb.p?PyObject_CallMethod(qb.p,"view","iiii",1,step,64,512):nullptr);
  Owned squares(heads.p?PyObject_CallMethod(heads.p,"square",nullptr):nullptr);
  Owned mean(squares.p?PyObject_CallMethod(squares.p,"mean","iO",-1,Py_True):nullptr);
  Owned variance(mean.p?PyObject_CallMethod(mean.p,"add","d",1e-6):nullptr);
  Owned inverse(variance.p?PyObject_CallMethod(variance.p,"rsqrt",nullptr):nullptr);
  Owned qh(inverse.p?PyObject_CallMethod(heads.p,"mul","O",inverse.p):nullptr);
  if(!qh.p){std::fprintf(stderr,"original query failure row=%d stage=q-head\n",row);return -1;}
  Owned kv(PyObject_CallObject(linear.p,kv_args.p));
  if(!kv.p){std::fprintf(stderr,"original query failure row=%d stage=kv\n",row);return -1;}
  Owned kn(kv.p?learned_norm(model.p,kv.p,kweight.p,512):nullptr);
  if(!qa.p||!qn.p||!qb.p||!qh.p||!kv.p||!kn.p)return -1;
  PyObject *values[6]={qa.p,qn.p,qb.p,qh.p,kv.p,kn.p};
  for(int i=0;i<6;++i) {
    Owned flat(PyObject_CallMethod(values[i],"view","ii",step,widths[i]));
    if(!flat.p)return -1;
    PyList_SET_ITEM(pieces[i].p,row/step,flat.p);flat.p=nullptr;
  }
  }
  if(!save(x.p,u8.p,std::string(output)+"/input.bf16",int64_t(rows)*4096*2)||
    !save(qweight.p,u8.p,std::string(output)+"/q_norm.bf16",1024*2)||
    !save(kweight.p,u8.p,std::string(output)+"/kv_norm.bf16",512*2))return -1;
  for(int i=0;i<6;++i) {
    Owned joined(PyObject_CallMethod(torch.p,"cat","Oi",pieces[i].p,0));
    if(!joined.p||!save(joined.p,u8.p,std::string(output)+"/"+stages[i]+".bf16",int64_t(rows)*widths[i]*2))return -1;
  }
  std::printf("original_query_kv_complete layer=%d rows=%d submission_rows=%d original_fp8_kernels=true original_learned_norm=true numerical_head_boundary=bf16 tp=1 full_model=false\n",layer,rows,step);
  return 0;
}

int main(int argc,char **argv) {
  if(argc!=6&&argc!=7&&argc!=8)return 2;
  char *end=nullptr;long rows=std::strtol(argv[5],&end,10);
  if(end==argv[5]||!end||*end||rows<1||rows>144)return 2;
  int layer=0;
  if(argc>=7){end=nullptr;long value=std::strtol(argv[6],&end,10);if(end==argv[6]||!end||*end||value<0||value>=43)return 2;layer=static_cast<int>(value);}
  if(argc==8&&std::string(argv[7])!="--one-row")return 2;
  Py_Initialize();const int result=execute(argv[1],argv[2],argv[3],argv[4],static_cast<int>(rows),layer,argc==8);
  if(PyErr_Occurred())PyErr_Print();
  return Py_FinalizeEx()<0||result?1:0;
}
