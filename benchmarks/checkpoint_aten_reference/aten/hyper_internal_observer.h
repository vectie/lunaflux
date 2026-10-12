#pragma once
// Offline return observer: literal intermediates from unmodified Block.hc_pre.
#include "original_tensor_io.h"
#include <cstring>
#include <map>

static std::map<std::string,std::vector<char>> hyper_internal_values;
static PyObject *hyper_internal_u8=nullptr;
static bool hyper_norm_observer=false;

static int append_hyper_internal(const char *name,PyObject *value,int width) {
  Owned cpu(PyObject_CallMethod(value,"cpu",nullptr));
  Owned contiguous(cpu.p?PyObject_CallMethod(cpu.p,"contiguous",nullptr):nullptr);
  Owned raw(contiguous.p?PyObject_CallMethod(contiguous.p,"view","O",hyper_internal_u8):nullptr);
  Owned array(raw.p?PyObject_CallMethod(raw.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p||PyBytes_Size(bytes.p)!=width*4){
    PyErr_SetString(PyExc_ValueError,"original hyper requires one complete F32 row");return -1;
  }
  auto &all=hyper_internal_values[name];const char *begin=PyBytes_AsString(bytes.p);
  all.insert(all.end(),begin,begin+width*4);return 0;
}

static int trace_hyper_internal(PyObject *,PyFrameObject *frame,int event,PyObject *) {
  if(event!=PyTrace_RETURN||!hyper_internal_u8)return 0;
  Owned code(reinterpret_cast<PyObject *>(PyFrame_GetCode(frame)));
  Owned name(code.p?PyObject_GetAttrString(code.p,"co_name"):nullptr);
  if(!name.p)return -1;
  const char *function=PyUnicode_AsUTF8(name.p);
  if(!function)return -1;
  const bool norm=hyper_norm_observer&&std::strcmp(function,"forward")==0;
  if(std::strcmp(function,"hc_pre")!=0&&!norm)return 0;
  Owned locals(PyFrame_GetLocals(frame));
  if(!locals.p)return -1;
  if(norm) {
    auto *variance=PyDict_GetItemString(locals.p,"var");
    if(!variance){PyErr_SetString(PyExc_ValueError,"original RMS variance missing");return -1;}
    return append_hyper_internal("norm-variance",variance,1);
  }
  // hc_pre keeps no named variance. Replay the exact installed square/mean
  // operation on its actual flattened operand; rsqrt remains the observed one.
  // This additional diagnostic operation never feeds back into original math.
  auto *operand=PyDict_GetItemString(locals.p,"x");
  Owned squared(operand?PyObject_CallMethod(operand,"square",nullptr):nullptr);
  Owned variance(squared.p?PyObject_CallMethod(squared.p,"mean","iO",-1,Py_True):nullptr);
  if(!variance.p||append_hyper_internal("control-mean-replay",variance.p,1)<0)return -1;
  for(const auto &capture: {std::pair{"mixes",24},std::pair{"rsqrt",1}}) {
    auto *value=PyDict_GetItemString(locals.p,capture.first);
    if(!value){PyErr_SetString(PyExc_ValueError,"original hc_pre intermediate missing");return -1;}
    if(append_hyper_internal(capture.first,value,capture.second)<0)return -1;
  }
  return 0;
}

static bool write_hyper_internals(const char *root,int rows) {
  for(const auto &capture: {std::pair{"mixes",24},std::pair{"rsqrt",1},std::pair{"control-mean-replay",1},std::pair{"norm-variance",1}}) {
    const auto &values=hyper_internal_values[capture.first];
    if(values.size()!=size_t(rows)*capture.second*4){
      PyErr_SetString(PyExc_ValueError,"original hc_pre internal row count differs");return false;
    }
    const std::string path=std::string(root)+"/"+capture.first+".f32";
    auto *file=std::fopen(path.c_str(),"wbx");if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,path.c_str());return false;}
    const bool wrote=std::fwrite(values.data(),1,values.size(),file)==values.size();
    const bool closed=std::fclose(file)==0;if(!wrote||!closed)return false;
  }
  return true;
}
