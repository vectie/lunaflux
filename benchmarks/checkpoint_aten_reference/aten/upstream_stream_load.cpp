// Test-only allocation adapter for the original safetensors load_model API.
// Same named tensors/copy_ arithmetic, one CPU tensor at a time. Never linked
// into production. GB10 unified RAM cannot hold an 80-GB GPU model AND an
// 80-GB CPU state dictionary simultaneously.
#include <Python.h>
#include <cstdio>

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *v) : p(v) {}
  ~Owned() { Py_XDECREF(p); }
  Owned(const Owned &) = delete;
};

static PyObject *stream_load(PyObject *, PyObject *args, PyObject *kwargs) {
  PyObject *model=nullptr,*filename=nullptr,*device=nullptr;
  int strict=1; const char *backend="mmap";
  static const char *names[]={"model","filename","strict","device","backend",nullptr};
  if(!PyArg_ParseTupleAndKeywords(args,kwargs,"OO|pO$s:load_model",
      const_cast<char **>(names),&model,&filename,&strict,&device,&backend)) return nullptr;
  Owned safe(PyImport_ImportModule("safetensors"));
  Owned torch_safe(PyImport_ImportModule("safetensors.torch"));
  Owned open(safe.p?PyObject_GetAttrString(safe.p,"safe_open"):nullptr);
  Owned positional(PyTuple_Pack(1,filename));
  // Keep the original mmap backend, but release each mapped CPU tensor after
  // its blocking copy. There is no complete owning CPU state dictionary.
  Owned options(Py_BuildValue("{s:s,s:s,s:s}","framework","pt","device","cpu","backend",backend));
  Owned file(open.p?PyObject_Call(open.p,positional.p,options.p):nullptr);
  Owned state(PyObject_CallMethod(model,"state_dict",nullptr));
  Owned keys(file.p?PyObject_CallMethod(file.p,"keys",nullptr):nullptr);
  Owned missing(state.p?PySet_New(state.p):nullptr);
  Owned unexpected(PyList_New(0));
  if(!file.p||!state.p||!keys.p||!missing.p||!unexpected.p||!torch_safe.p) return nullptr;
  Owned remove(PyObject_GetAttrString(torch_safe.p,"_remove_duplicate_names"));
  Owned remove_args(PyTuple_Pack(1,state.p));
  Owned remove_options(Py_BuildValue("{s:O}","preferred_names",keys.p));
  Owned aliases(remove.p?PyObject_Call(remove.p,remove_args.p,remove_options.p):nullptr);
  if(!aliases.p) return nullptr;
  Py_ssize_t loaded=0;
  for(Py_ssize_t i=0;i<PyList_Size(keys.p);++i) {
    auto *key=PyList_GetItem(keys.p,i);
    auto *target=PyDict_GetItemWithError(state.p,key);
    if(!target) {
      if(PyErr_Occurred()||PyList_Append(unexpected.p,key)<0) return nullptr;
      continue;
    }
    Owned value(PyObject_CallMethod(file.p,"get_tensor","O",key));
    Owned source_shape(value.p?PyObject_GetAttrString(value.p,"shape"):nullptr);
    Owned target_shape(PyObject_GetAttrString(target,"shape"));
    if(!source_shape.p||!target_shape.p) {
      std::fprintf(stderr,"stream_load_shape_failure key=%s source_type=%s target_type=%s\n",
        PyUnicode_AsUTF8(key),value.p?Py_TYPE(value.p)->tp_name:"null",Py_TYPE(target)->tp_name);
      return nullptr;
    }
    if(PyObject_RichCompareBool(source_shape.p,target_shape.p,Py_EQ)!=1) {
      PyErr_Format(PyExc_ValueError,"original parameter shape mismatch: %U",key);
      return nullptr;
    }
    Owned copied(PyObject_CallMethod(target,"copy_","O",value.p));
    if(!copied.p||PySet_Discard(missing.p,key)<0) return nullptr;
    ++loaded;
    if(loaded%512==0) { std::printf("upstream_stream_loaded=%zd\n",loaded); std::fflush(stdout); }
  }
  // Preserve the upstream duplicate-parameter missing/unexpected semantics.
  PyObject *ignored,*group; Py_ssize_t at=0;
  while(PyDict_Next(aliases.p,&at,&ignored,&group)) {
    for(Py_ssize_t i=0;i<PyList_Size(group);++i) {
      auto *key=PyList_GetItem(group,i);
      const int removed=PySet_Discard(missing.p,key);
      if(removed<0||(removed==0&&PyList_Append(unexpected.p,key)<0)) return nullptr;
    }
  }
  if(strict&&(PySet_Size(missing.p)||PyList_Size(unexpected.p))) {
    PyErr_Format(PyExc_RuntimeError,"stream load_model missing=%zd unexpected=%zd",
      PySet_Size(missing.p),PyList_Size(unexpected.p)); return nullptr;
  }
  std::printf("upstream_stream_load_complete tensors=%zd missing=%zd unexpected=%zd arithmetic_changed=false\n",
    loaded,PySet_Size(missing.p),PyList_Size(unexpected.p)); std::fflush(stdout);
  return PyTuple_Pack(2,missing.p,unexpected.p);
}

static PyMethodDef method={"load_model",reinterpret_cast<PyCFunction>(stream_load),
  METH_VARARGS|METH_KEYWORDS,"Allocation-only streaming original checkpoint loader."};

int install_upstream_stream_loader() {
  Owned module(PyImport_ImportModule("safetensors.torch"));
  Owned function(PyCFunction_NewEx(&method,nullptr,nullptr));
  if(!module.p||!function.p) return -1;
  return PyObject_SetAttrString(module.p,"load_model",function.p);
}
