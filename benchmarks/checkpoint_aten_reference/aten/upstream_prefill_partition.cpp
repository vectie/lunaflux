// Explicit test-only shape ablation. Every chunk invokes the original
// Transformer.forward; no attention, projection, logits or sampling math is
// reimplemented here. The unmodified whole-prompt path is the default.
#include <Python.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct Partition { PyObject *original; int rows; };
struct Owned {
  PyObject *p;
  explicit Owned(PyObject *v):p(v){}
  ~Owned(){Py_XDECREF(p);}
};
static void release(PyObject *capsule) {
  auto *state=static_cast<Partition*>(PyCapsule_GetPointer(capsule,"original.prefill.partition"));
  if(state){Py_DECREF(state->original);delete state;}
}
static PyObject *partition(PyObject *capsule,PyObject *args,PyObject *kwargs) {
  auto *state=static_cast<Partition*>(PyCapsule_GetPointer(capsule,"original.prefill.partition"));
  PyObject *self=nullptr,*input=nullptr;int start=0;
  static const char *names[]={"self","input_ids","start_pos",nullptr};
  if(!state||!PyArg_ParseTupleAndKeywords(args,kwargs,"OO|i:forward",const_cast<char**>(names),&self,&input,&start))return nullptr;
  Owned count(PyObject_CallMethod(input,"size","i",1));
  if(!count.p)return nullptr;
  const auto rows=PyLong_AsLong(count.p);
  if(PyErr_Occurred())return nullptr;
  if(start!=0||rows<=state->rows)return PyObject_Call(state->original,args,kwargs);
  if(rows>256){PyErr_SetString(PyExc_ValueError,"prefill shape ablation exceeds bounded context");return nullptr;}
  if(setenv("LUNA_UPSTREAM_PREFILL_END",std::to_string(rows).c_str(),1)!=0){PyErr_SetFromErrno(PyExc_OSError);return nullptr;}
  struct CaptureEnd { ~CaptureEnd(){unsetenv("LUNA_UPSTREAM_PREFILL_END");} } capture_end;
  Owned result(nullptr);
  for(int first=0;first<rows;first+=state->rows) {
    const int length=first+state->rows>rows?rows-first:state->rows;
    Owned chunk(PyObject_CallMethod(input,"narrow","iii",1,first,length));
    Owned position(PyLong_FromLong(first));
    Owned call(chunk.p&&position.p?PyTuple_Pack(3,self,chunk.p,position.p):nullptr);
    if(!call.p)return nullptr;
    auto *next=PyObject_CallObject(state->original,call.p);
    if(!next)return nullptr;
    Py_XDECREF(result.p);result.p=next;
  }
  std::printf("original_prefill_partition rows=%ld chunk_rows=%d original_forward_calls=%ld arithmetic_source_changed=false shape_ablation=true\n",rows,state->rows,(rows+state->rows-1)/state->rows);
  return Py_NewRef(result.p);
}
static PyMethodDef method={"forward",reinterpret_cast<PyCFunction>(partition),METH_VARARGS|METH_KEYWORDS,"Explicit original-forward prefill shape ablation."};
int install_upstream_prefill_partition() {
  const char *option=std::getenv("LUNA_UPSTREAM_PREFILL_ROWS");
  if(!option||!*option)return 0;
  char *end=nullptr;long rows=std::strtol(option,&end,10);
  // Original get_window_topk_idxs(start_pos>0) broadcasts one decode row's
  // read set. Multirow chunks would change causal semantics, not just rounding.
  if(!end||*end||rows!=1){PyErr_SetString(PyExc_ValueError,"original continuation supports only one-row prefill increments");return -1;}
  Owned module(PyImport_ImportModule("model"));
  Owned cls(module.p?PyObject_GetAttrString(module.p,"Transformer"):nullptr);
  Owned original(cls.p?PyObject_GetAttrString(cls.p,"forward"):nullptr);
  if(!original.p)return -1;
  auto *state=new Partition{Py_NewRef(original.p),static_cast<int>(rows)};
  Owned capsule(PyCapsule_New(state,"original.prefill.partition",release));
  if(!capsule.p){Py_DECREF(state->original);delete state;return -1;}
  Owned callback(PyCFunction_NewEx(&method,capsule.p,nullptr));
  Owned descriptor(callback.p?PyInstanceMethod_New(callback.p):nullptr);
  if(!descriptor.p)return -1;
  return PyObject_SetAttrString(cls.p,"forward",descriptor.p);
}

static std::vector<std::pair<int,int>> probe_calls;
static PyObject *probe_forward(PyObject *,PyObject *args) {
  PyObject *self=nullptr,*input=nullptr;int start=0;
  if(!PyArg_ParseTuple(args,"OOi",&self,&input,&start))return nullptr;
  Owned size(PyObject_CallMethod(input,"size","i",1));
  if(!size.p)return nullptr;
  int rows=static_cast<int>(PyLong_AsLong(size.p));
  if(PyErr_Occurred())return nullptr;
  probe_calls.emplace_back(start,rows);
  return PyLong_FromLong(start*100+rows);
}
static PyMethodDef probe_method={"forward",probe_forward,METH_VARARGS,"CPU-only invocation probe, not a mathematical reference."};
int upstream_prefill_partition_probe() {
  Owned callback(PyCFunction_NewEx(&probe_method,nullptr,nullptr));
  Owned dictionary(Py_BuildValue("{s:O}","forward",callback.p));
  Owned name(PyUnicode_FromString("Transformer"));
  Owned bases(PyTuple_Pack(1,reinterpret_cast<PyObject*>(&PyBaseObject_Type)));
  Owned args(PyTuple_Pack(3,name.p,bases.p,dictionary.p));
  Owned cls(PyObject_CallObject(reinterpret_cast<PyObject*>(&PyType_Type),args.p));
  Owned module(PyModule_New("model"));
  if(!cls.p||!module.p||PyObject_SetAttrString(module.p,"Transformer",cls.p)<0||PyDict_SetItemString(PyImport_GetModuleDict(),"model",module.p)<0)return -1;
  if(setenv("LUNA_UPSTREAM_PREFILL_ROWS","1",1)!=0||install_upstream_prefill_partition()<0)return -1;
  Owned self(PyObject_CallNoArgs(cls.p)),torch(PyImport_ImportModule("torch"));
  Owned zeros(torch.p?PyObject_GetAttrString(torch.p,"zeros"):nullptr);
  Owned shape(Py_BuildValue("(ii)",1,13)),zero_args(PyTuple_Pack(1,shape.p));
  Owned options(Py_BuildValue("{s:s}","device","cpu"));
  Owned input(zeros.p?PyObject_Call(zeros.p,zero_args.p,options.p):nullptr);
  Owned result(input.p&&self.p?PyObject_CallMethod(self.p,"forward","Oi",input.p,0):nullptr);
  std::vector<std::pair<int,int>> expected;
  for(int row=0;row<13;++row)expected.emplace_back(row,1);
  if(!result.p||PyLong_AsLong(result.p)!=1201||probe_calls!=expected||std::getenv("LUNA_UPSTREAM_PREFILL_END")) {
    if(!PyErr_Occurred())PyErr_SetString(PyExc_AssertionError,"partition binding, final chunk or capture scope mismatch");
    return -1;
  }
  Owned decode(PyObject_CallMethod(input.p,"narrow","iii",1,0,5));
  Owned next(decode.p?PyObject_CallMethod(self.p,"forward","Oi",decode.p,13):nullptr);
  if(!next.p||PyLong_AsLong(next.p)!=1305||probe_calls.size()!=14||probe_calls.back()!=std::make_pair(13,5)) {
    if(!PyErr_Occurred())PyErr_SetString(PyExc_AssertionError,"decode was repartitioned");return -1;
  }
  unsetenv("LUNA_UPSTREAM_PREFILL_ROWS");
  std::puts("original_prefill_partition_probe_passed binding=true tail=true decode_unchanged=true capture_scope_retired=true gpu_work=false");
  return 0;
}
