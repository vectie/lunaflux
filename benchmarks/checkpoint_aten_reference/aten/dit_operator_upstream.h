#pragma once
#include "refiner_operator_upstream.h"

// Each operation consumes its literal native predecessor. Do not propagate a
// reference attention result into projection and call that a local comparison.
inline bool dit_operators(PyObject *torch,PyObject *bf16,PyObject *u8,
                          PyObject *block,PyObject *documents,int maximum,
                          const char *input,const char *output,int rows) {
  auto read=[&](const char *name,int width,int count) {
    return refiner_input(torch,bf16,input,std::string("block-0-")+name,count,width);
  };
  auto write=[&](const char *name,PyObject *value,int width) {
    return value&&save(value,u8,std::string(output)+"/block-0-"+name+".bf16",
                       int64_t(rows)*width*2);
  };
  Owned attention(PyObject_GetAttrString(block,"attn"));
  Owned heads_value(attention.p?PyObject_GetAttrString(attention.p,"num_heads"):nullptr);
  Owned dimension_value(attention.p?PyObject_GetAttrString(attention.p,"head_dim"):nullptr);
  if(!heads_value.p||!dimension_value.p)return false;
  const int heads=static_cast<int>(PyLong_AsLong(heads_value.p));
  const int dimension=static_cast<int>(PyLong_AsLong(dimension_value.p));
  const int inner=heads*dimension;
  if(PyErr_Occurred()||heads<=0||dimension<=0||inner>65536)return false;
  Owned qkv(read("qk-norm",inner,rows*3));
  Owned q0(qkv.p?PyObject_CallMethod(qkv.p,"narrow","iii",0,0,rows):nullptr);
  Owned k0(qkv.p?PyObject_CallMethod(qkv.p,"narrow","iii",0,rows,rows):nullptr);
  Owned v0(qkv.p?PyObject_CallMethod(qkv.p,"narrow","iii",0,2*rows,rows):nullptr);
  Owned q(q0.p?PyObject_CallMethod(q0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned k(k0.p?PyObject_CallMethod(k0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned v(v0.p?PyObject_CallMethod(v0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned packed(attention.p?PyObject_GetAttrString(attention.p,"_run_packed_attention"):nullptr);
  Owned args(q.p&&k.p&&v.p?PyTuple_Pack(3,q.p,k.p,v.p):nullptr);
  Owned kw(Py_BuildValue("{s:O,s:i}","cu_seqlens",documents,"max_seqlen",maximum));
  Owned attended(packed.p&&args.p?PyObject_Call(packed.p,args.p,kw.p):nullptr);
  if(!write("attention",attended.p,inner))return false;
  Owned native_attention(read("attention",inner,rows));
  Owned projection(attention.p?PyObject_GetAttrString(attention.p,"out_proj"):nullptr);
  Owned projected(refiner_call(projection.p,native_attention.p,true));
  if(!write("projection",projected.p,5376))return false;
  Owned mlp(PyObject_GetAttrString(block,"mlp"));
  Owned fc1(mlp.p?PyObject_GetAttrString(mlp.p,"fc1"):nullptr);
  Owned mlp_input(read("mlp-input",5376,rows));
  Owned hidden(refiner_call(fc1.p,mlp_input.p,true));
  Owned halves(hidden.p?PyObject_CallMethod(hidden.p,"chunk","ii",2,1):nullptr);
  Owned functional(PyImport_ImportModule("torch.nn.functional"));
  Owned silu(functional.p?PyObject_GetAttrString(functional.p,"silu"):nullptr);
  Owned activated(halves.p?refiner_call(silu.p,PyTuple_GetItem(halves.p,0)):nullptr);
  Owned product(activated.p&&halves.p?PyNumber_Multiply(activated.p,PyTuple_GetItem(halves.p,1)):nullptr);
  Owned size(product.p?PyObject_CallMethod(product.p,"size","i",1):nullptr);
  if(!size.p||PyLong_AsLong(size.p)!=14336||!write("gate-up",product.p,14336))return false;
  Owned native_product(read("gate-up",14336,rows));
  Owned fc2(mlp.p?PyObject_GetAttrString(mlp.p,"fc2"):nullptr);
  Owned down(refiner_call(fc2.p,native_product.p,true));
  return write("down",down.p,5376);
}
