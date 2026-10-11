#pragma once
#include "original_tensor_io.h"

// Offline literal interventions. Every operation is called on its installed
// original module/function; this adapter supplies layout/I/O, not arithmetic.
inline PyObject *refiner_input(PyObject *torch,PyObject *bf16,const char *root,
                              const std::string &name,int rows,int width) {
  Owned tensor(literal_bf16(torch,bf16,(std::string(root)+"/"+name+".bf16").c_str(),rows,width));
  return tensor.p?PyObject_CallMethod(tensor.p,"view","ii",rows,width):nullptr;
}

inline PyObject *refiner_call(PyObject *module,PyObject *input,bool pair=false) {
  Owned args(input?PyTuple_Pack(1,input):nullptr);
  Owned output(module&&args.p?PyObject_CallObject(module,args.p):nullptr);
  if(!output.p)return nullptr;
  return Py_NewRef(pair?PyTuple_GetItem(output.p,0):output.p);
}

inline PyObject *refiner_cat(PyObject *torch,PyObject *q,PyObject *k,PyObject *v) {
  Owned parts(q&&k&&v?PyTuple_Pack(3,q,k,v):nullptr);
  Owned fn(PyObject_GetAttrString(torch,"cat")),args(parts.p?PyTuple_Pack(1,parts.p):nullptr);
  return fn.p&&args.p?PyObject_Call(fn.p,args.p,nullptr):nullptr;
}

inline bool refiner_operators(PyObject *torch,PyObject *bf16,PyObject *u8,
                              PyObject *block,PyObject *attention_kw,
                              const char *input,const char *out,int layer,int rows) {
  const std::string prefix="refiner-"+std::to_string(layer)+"-";
  auto read=[&](const std::string &name,int width=5376,int count=0) {
    return refiner_input(torch,bf16,input,name,count?count:rows,width);
  };
  auto write=[&](const char *name,PyObject *value,int width=5376,int count=0) {
    return value&&save(value,u8,std::string(out)+"/"+prefix+name+".bf16",int64_t(count?count:rows)*width*2);
  };
  Owned state(read(layer?"refiner-0":"context"));
  Owned norm1(PyObject_GetAttrString(block,"norm1")),norm2(PyObject_GetAttrString(block,"norm2"));
  Owned n1(refiner_call(norm1.p,state.p));
  if(!write("norm1",n1.p))return false;
  Owned att(PyObject_GetAttrString(block,"attn")),qkv_proj(att.p?PyObject_GetAttrString(att.p,"qkv_proj"):nullptr);
  Owned heads_value(att.p?PyObject_GetAttrString(att.p,"num_heads"):nullptr);
  Owned dimension_value(att.p?PyObject_GetAttrString(att.p,"head_dim"):nullptr);
  if(!heads_value.p||!dimension_value.p)return false;
  const int heads=static_cast<int>(PyLong_AsLong(heads_value.p));
  const int dimension=static_cast<int>(PyLong_AsLong(dimension_value.p));
  const int inner=heads*dimension;
  if(PyErr_Occurred()||inner<=0)return false;
  Owned native_n1(read(prefix+"norm1")),qkv(refiner_call(qkv_proj.p,native_n1.p,true));
  Owned pieces(qkv.p?PyObject_CallMethod(qkv.p,"chunk","ii",3,1):nullptr);
  auto *q0=pieces.p?PyTuple_GetItem(pieces.p,0):nullptr;
  auto *k0=pieces.p?PyTuple_GetItem(pieces.p,1):nullptr;
  auto *v0=pieces.p?PyTuple_GetItem(pieces.p,2):nullptr;
  Owned planar(refiner_cat(torch,q0,k0,v0));
  if(!write("qkv",planar.p,inner,rows*3))return false;
  // Native scratch is planar Q,K,V. The original norm/attention uses THD.
  Owned native_qkv(read(prefix+"qkv",inner,rows*3));
  Owned nq(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,0,rows):nullptr);
  Owned nk(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,rows,rows):nullptr);
  Owned nv(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,rows*2,rows):nullptr);
  Owned q(nq.p?PyObject_CallMethod(nq.p,"view","iii",rows,heads,dimension):nullptr);
  Owned k(nk.p?PyObject_CallMethod(nk.p,"view","iii",rows,heads,dimension):nullptr);
  Owned qnorm(att.p?PyObject_GetAttrString(att.p,"q_norm"):nullptr),knorm(att.p?PyObject_GetAttrString(att.p,"k_norm"):nullptr);
  Owned normalized_q(refiner_call(qnorm.p,q.p)),normalized_k(refiner_call(knorm.p,k.p));
  Owned qflat(normalized_q.p?PyObject_CallMethod(normalized_q.p,"view","ii",rows,inner):nullptr);
  Owned kflat(normalized_k.p?PyObject_CallMethod(normalized_k.p,"view","ii",rows,inner):nullptr);
  Owned normalized(refiner_cat(torch,qflat.p,kflat.p,nv.p));
  if(!write("qk-norm",normalized.p,inner,rows*3))return false;
  Owned native_norm(read(prefix+"qk-norm",inner,rows*3));
  Owned aq0(native_norm.p?PyObject_CallMethod(native_norm.p,"narrow","iii",0,0,rows):nullptr);
  Owned ak0(native_norm.p?PyObject_CallMethod(native_norm.p,"narrow","iii",0,rows,rows):nullptr);
  Owned av0(native_norm.p?PyObject_CallMethod(native_norm.p,"narrow","iii",0,rows*2,rows):nullptr);
  Owned aq(aq0.p?PyObject_CallMethod(aq0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned ak(ak0.p?PyObject_CallMethod(ak0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned av(av0.p?PyObject_CallMethod(av0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned packed(att.p?PyObject_GetAttrString(att.p,"_run_packed_attention"):nullptr);
  Owned attargs(aq.p&&ak.p&&av.p?PyTuple_Pack(3,aq.p,ak.p,av.p):nullptr);
  Owned a(packed.p&&attargs.p?PyObject_Call(packed.p,attargs.p,attention_kw):nullptr);
  if(!write("attention",a.p,inner))return false;
  Owned native_a(read(prefix+"attention",inner)),proj(att.p?PyObject_GetAttrString(att.p,"out_proj"):nullptr);
  Owned projected(refiner_call(proj.p,native_a.p,true));
  if(!write("projection",projected.p))return false;
  Owned native_p(read(prefix+"projection"));
  Owned residual(state.p&&native_p.p?PyNumber_Add(state.p,native_p.p):nullptr);
  if(!write("attention-residual",residual.p))return false;
  Owned native_r(read(prefix+"attention-residual")),n2(refiner_call(norm2.p,native_r.p));
  if(!write("norm2",n2.p))return false;
  Owned mlp(PyObject_GetAttrString(block,"mlp")),fc1(mlp.p?PyObject_GetAttrString(mlp.p,"fc1"):nullptr);
  Owned native_n2(read(prefix+"norm2")),hidden(refiner_call(fc1.p,native_n2.p,true));
  Owned halves(hidden.p?PyObject_CallMethod(hidden.p,"chunk","ii",2,1):nullptr);
  Owned functional(PyImport_ImportModule("torch.nn.functional"));
  Owned silu(functional.p?PyObject_GetAttrString(functional.p,"silu"):nullptr);
  Owned gate(halves.p?refiner_call(silu.p,PyTuple_GetItem(halves.p,0)):nullptr);
  Owned product(gate.p&&halves.p?PyNumber_Multiply(gate.p,PyTuple_GetItem(halves.p,1)):nullptr);
  if(!write("gate-up",product.p,14336))return false;
  Owned native_product(read(prefix+"gate-up",14336)),fc2(mlp.p?PyObject_GetAttrString(mlp.p,"fc2"):nullptr);
  Owned down(refiner_call(fc2.p,native_product.p,true));
  if(!write("down",down.p))return false;
  Owned native_down(read(prefix+"down"));
  Owned final(native_r.p&&native_down.p?PyNumber_Add(native_r.p,native_down.p):nullptr);
  return final.p&&save(final.p,u8,std::string(out)+"/refiner-"+std::to_string(layer)+".bf16",int64_t(rows)*5376*2);
}
