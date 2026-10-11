#pragma once
// Offline observer only. PyTrace_RETURN preserves the original Block math,
// literal input IDs, one-row geometry and persistent KV state.
#include "original_tensor_io.h"
#include <cstring>
#include <map>

struct OriginalBoundaries {
  PyObject *attention_norm=nullptr, *attention=nullptr, *ffn_norm=nullptr, *ffn=nullptr;
  PyObject *block=nullptr, *u8=nullptr;
  int post_count=0;
  std::map<std::string,std::vector<char>> values;
};
static OriginalBoundaries *boundary_observer=nullptr;

static int append_original_boundary(const char *name,PyObject *value,int width) {
  Owned cpu(PyObject_CallMethod(value,"cpu",nullptr));
  Owned contiguous(cpu.p?PyObject_CallMethod(cpu.p,"contiguous",nullptr):nullptr);
  Owned raw(contiguous.p?PyObject_CallMethod(contiguous.p,"view","O",boundary_observer->u8):nullptr);
  Owned array(raw.p?PyObject_CallMethod(raw.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p||PyBytes_Size(bytes.p)!=width*2){
    PyErr_SetString(PyExc_ValueError,"original boundary must retain one complete BF16 row");return -1;
  }
  const char *begin=PyBytes_AsString(bytes.p);
  auto &values=boundary_observer->values[name];
  values.insert(values.end(),begin,begin+width*2);
  return 0;
}

static int trace_original_boundary(PyObject *,PyFrameObject *frame,int event,PyObject *result) {
  if(event!=PyTrace_RETURN||result==Py_None||!boundary_observer)return 0;
  Owned code(reinterpret_cast<PyObject *>(PyFrame_GetCode(frame)));
  Owned name(code.p?PyObject_GetAttrString(code.p,"co_name"):nullptr);
  Owned locals(PyFrame_GetLocals(frame));
  if(!name.p||!locals.p)return -1;
  const char *function=PyUnicode_AsUTF8(name.p);
  auto *self=PyDict_GetItemString(locals.p,"self");
  if(!function||!self)return PyErr_Occurred()?-1:0;
  if(std::strcmp(function,"forward")==0) {
    for(auto boundary : {
      std::pair{"attention-input",boundary_observer->attention_norm},
      std::pair{"attention-output",boundary_observer->attention},
      std::pair{"ffn-input",boundary_observer->ffn_norm},
      std::pair{"ffn-output",boundary_observer->ffn}
    })if(self==boundary.second)return append_original_boundary(boundary.first,result,4096);
  }
  if(self==boundary_observer->block&&std::strcmp(function,"hc_post")==0&&
     boundary_observer->post_count++==0)
    return append_original_boundary("attention-residual",result,16384);
  return 0;
}

static bool write_original_boundaries(const OriginalBoundaries &observer,const char *out) {
  if(observer.values.size()!=5){PyErr_SetString(PyExc_ValueError,"original boundary observer incomplete");return false;}
  for(const auto &[name,values]:observer.values) {
    const size_t expected=144ULL*(name=="attention-residual"?16384:4096)*2;
    if(values.size()!=expected){PyErr_SetString(PyExc_ValueError,"original boundary row count differs");return false;}
    const std::string path=std::string(out)+"/"+name+".bf16";
    auto *file=std::fopen(path.c_str(),"wbx");
    if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,path.c_str());return false;}
    const bool written=std::fwrite(values.data(),1,values.size(),file)==values.size();
    const bool closed=std::fclose(file)==0;
    if(!written||!closed){PyErr_SetString(PyExc_OSError,"original boundary write failed");return false;}
  }
  return true;
}
