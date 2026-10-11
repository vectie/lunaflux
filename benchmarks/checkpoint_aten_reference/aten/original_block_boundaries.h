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
  bool inside_attention=false;
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
  if(!boundary_observer||(event!=PyTrace_RETURN&&event!=PyTrace_CALL))return 0;
  Owned code(reinterpret_cast<PyObject *>(PyFrame_GetCode(frame)));
  Owned name(code.p?PyObject_GetAttrString(code.p,"co_name"):nullptr);
  Owned locals(PyFrame_GetLocals(frame));
  if(!name.p||!locals.p)return -1;
  const char *function=PyUnicode_AsUTF8(name.p);
  auto *self=PyDict_GetItemString(locals.p,"self");
  if(!function)return -1;
  if(self==boundary_observer->attention&&std::strcmp(function,"forward")==0&&event==PyTrace_CALL) {
    boundary_observer->inside_attention=true;return 0;
  }
  if(event!=PyTrace_RETURN||result==Py_None)return 0;
  if(boundary_observer->inside_attention&&std::strcmp(function,"apply_rotary_emb")==0&&
     PyDict_GetItemString(locals.p,"inverse")==Py_True) {
    // Observe the original in-place return before grouped output projection.
    // Prefix channels outside RoPE are retained in the same complete row.
    Owned caller(reinterpret_cast<PyObject *>(PyFrame_GetBack(frame)));
    Owned caller_locals(caller.p?PyFrame_GetLocals(reinterpret_cast<PyFrameObject *>(caller.p)):nullptr);
    auto *full=caller_locals.p?PyDict_GetItemString(caller_locals.p,"o"):nullptr;
    if(!full){PyErr_SetString(PyExc_ValueError,"original inverse caller output missing");return -1;}
    return append_original_boundary("attention-inverse",full,32768);
  }
  if(boundary_observer->inside_attention&&std::strcmp(function,"sparse_attn")==0) {
    // The original recursively partitions heads on GB10. Retain its complete
    // outer return, not the four 16-head internal returns as extra rows.
    auto *query=PyDict_GetItemString(locals.p,"q");
    Owned count(query?PyObject_CallMethod(query,"numel",nullptr):nullptr);
    if(!count.p)return -1;
    if(PyLong_AsLong(count.p)==32768)
      return append_original_boundary("sparse-attention",result,32768);
    return PyErr_Occurred()?-1:0;
  }
  if(!self)return 0;
  if(self==boundary_observer->attention&&std::strcmp(function,"forward")==0) {
    boundary_observer->inside_attention=false;
    for(const auto &component:std::vector<std::pair<const char*,int>>{
      {"qr",1024},{"q",32768},{"kv",512},{"o",8192}}) {
      const char *name=std::strcmp(component.first,"qr")==0?"attention-low-rank":
        std::strcmp(component.first,"q")==0?"q-rope":
        std::strcmp(component.first,"kv")==0?"kv-sim":"attention-output-a";
      auto *value=PyDict_GetItemString(locals.p,component.first);
      if(!value||append_original_boundary(name,value,component.second)<0)return -1;
    }
  }
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
  if(observer.values.size()!=11){PyErr_SetString(PyExc_ValueError,"original boundary observer incomplete");return false;}
  for(const auto &[name,values]:observer.values) {
    const int width=name=="attention-residual"?16384:
      (name=="q-rope"||name=="sparse-attention"||name=="attention-inverse")?32768:
      name=="kv-sim"?512:name=="attention-low-rank"?1024:
      name=="attention-output-a"?8192:4096;
    const size_t expected=144ULL*width*2;
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
