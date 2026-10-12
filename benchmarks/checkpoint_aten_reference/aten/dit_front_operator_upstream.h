#pragma once
#include "refiner_operator_upstream.h"

// Offline original frontiers. Every local operation receives its native
// predecessor; original results are never silently propagated to the next.
inline bool dit_front_operators(PyObject *torch,PyObject *bf16,PyObject *u8,
    PyObject *module,PyObject *block,PyObject *time,PyObject *indices,
    PyObject *frequencies,const char *input,const char *output,int rows,int times) {
  auto read=[&](const char *name,int width,int count) {
    return refiner_input(torch,bf16,input,name,count,width);
  };
  auto write=[&](const char *name,PyObject *value,int width,int count) {
    return value&&save(value,u8,std::string(output)+"/block-0-"+name+".bf16",
                       int64_t(count)*width*2);
  };
  Owned x(read("packed-input",5376,rows));
  Owned norm1(PyObject_GetAttrString(block,"norm1"));
  Owned normalized(refiner_call(norm1.p,x.p));
  if(!write("norm1",normalized.p,5376,rows))return false;
  Owned adaln(PyObject_GetAttrString(block,"adaln_proj"));
  Owned projected(refiner_call(adaln.p,time));
  Owned concat(PyObject_GetAttrString(torch,"cat"));
  Owned pargs(projected.p?PyTuple_Pack(1,projected.p):nullptr);
  Owned parameters(concat.p&&pargs.p?PyObject_CallObject(concat.p,pargs.p):nullptr);
  if(!write("parameters",parameters.p,5376,times*18))return false;
  Owned native_parameters(read("block-0-parameters",5376,times*18));
  Owned native_norm1(read("block-0-norm1",5376,rows));
  Owned shift(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,0,times*3):nullptr);
  Owned scale(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,times*3,times*3):nullptr);
  Owned affine_fn(PyObject_GetAttrString(module,"_modulate_scale_shift"));
  Owned dtype_kw(Py_BuildValue("{s:O}","dtype",bf16));
  Owned affine_args(native_norm1.p&&shift.p&&scale.p?PyTuple_Pack(4,native_norm1.p,shift.p,scale.p,indices):nullptr);
  Owned affine(affine_fn.p&&affine_args.p?PyObject_Call(affine_fn.p,affine_args.p,dtype_kw.p):nullptr);
  if(!write("attention-input",affine.p,5376,rows))return false;
  Owned attention(PyObject_GetAttrString(block,"attn"));
  Owned heads_value(attention.p?PyObject_GetAttrString(attention.p,"num_heads"):nullptr);
  Owned dimension_value(attention.p?PyObject_GetAttrString(attention.p,"head_dim"):nullptr);
  if(!heads_value.p||!dimension_value.p)return false;
  const int heads=static_cast<int>(PyLong_AsLong(heads_value.p));
  const int dimension=static_cast<int>(PyLong_AsLong(dimension_value.p));
  const int inner=heads*dimension;
  if(PyErr_Occurred()||heads<=0||dimension<=0||inner>65536)return false;
  Owned native_affine(read("block-0-attention-input",5376,rows));
  Owned qkv_proj(attention.p?PyObject_GetAttrString(attention.p,"qkv_proj"):nullptr);
  Owned projected_qkv(refiner_call(qkv_proj.p,native_affine.p,true));
  Owned pieces(projected_qkv.p?PyObject_CallMethod(projected_qkv.p,"chunk","ii",3,1):nullptr);
  Owned planar(pieces.p?refiner_cat(torch,PyTuple_GetItem(pieces.p,0),PyTuple_GetItem(pieces.p,1),PyTuple_GetItem(pieces.p,2)):nullptr);
  if(!write("qkv",planar.p,inner,3*rows))return false;
  Owned native_qkv(read("block-0-qkv",inner,rows*3));
  Owned q0(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,0,rows):nullptr);
  Owned k0(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,rows,rows):nullptr);
  Owned v0(native_qkv.p?PyObject_CallMethod(native_qkv.p,"narrow","iii",0,2*rows,rows):nullptr);
  Owned q(q0.p?PyObject_CallMethod(q0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned k(k0.p?PyObject_CallMethod(k0.p,"view","iii",rows,heads,dimension):nullptr);
  Owned qnorm(attention.p?PyObject_GetAttrString(attention.p,"q_norm"):nullptr);
  Owned knorm(attention.p?PyObject_GetAttrString(attention.p,"k_norm"):nullptr);
  Owned nq(refiner_call(qnorm.p,q.p)),nk(refiner_call(knorm.p,k.p));
  Owned rotate(PyObject_GetAttrString(module,"_apply_rope"));
  Owned qargs(nq.p?PyTuple_Pack(2,nq.p,frequencies):nullptr),kargs(nk.p?PyTuple_Pack(2,nk.p,frequencies):nullptr);
  Owned qr(rotate.p&&qargs.p?PyObject_CallObject(rotate.p,qargs.p):nullptr);
  Owned kr(rotate.p&&kargs.p?PyObject_CallObject(rotate.p,kargs.p):nullptr);
  Owned qflat(qr.p?PyObject_CallMethod(qr.p,"reshape","ii",rows,inner):nullptr);
  Owned kflat(kr.p?PyObject_CallMethod(kr.p,"reshape","ii",rows,inner):nullptr);
  Owned normalized_qkv(refiner_cat(torch,qflat.p,kflat.p,v0.p));
  if(!write("qk-norm",normalized_qkv.p,inner,3*rows))return false;
  Owned prior(read("block-0-attention-residual",5376,rows));
  Owned norm2(PyObject_GetAttrString(block,"norm2"));
  Owned n2(refiner_call(norm2.p,prior.p));
  if(!write("norm2",n2.p,5376,rows))return false;
  Owned native_norm2(read("block-0-norm2",5376,rows));
  Owned mlp_shift(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,3*times*3,times*3):nullptr);
  Owned mlp_scale(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,4*times*3,times*3):nullptr);
  Owned mlp_args(native_norm2.p&&mlp_shift.p&&mlp_scale.p?PyTuple_Pack(4,native_norm2.p,mlp_shift.p,mlp_scale.p,indices):nullptr);
  Owned mlp_input(affine_fn.p&&mlp_args.p?PyObject_Call(affine_fn.p,mlp_args.p,dtype_kw.p):nullptr);
  if(!write("mlp-input",mlp_input.p,5376,rows))return false;
  Owned gate_fn(PyObject_GetAttrString(module,"_modulate_gate"));
  Owned native_projection(read("block-0-projection",5376,rows));
  Owned attention_gate(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,2*times*3,times*3):nullptr);
  Owned residual_args(x.p&&attention_gate.p&&native_projection.p?PyTuple_Pack(4,x.p,attention_gate.p,native_projection.p,indices):nullptr);
  Owned residual(gate_fn.p&&residual_args.p?PyObject_Call(gate_fn.p,residual_args.p,dtype_kw.p):nullptr);
  if(!write("attention-residual",residual.p,5376,rows))return false;
  Owned native_down(read("block-0-down",5376,rows));
  Owned mlp_gate(native_parameters.p?PyObject_CallMethod(native_parameters.p,"narrow","iii",0,5*times*3,times*3):nullptr);
  Owned final_args(prior.p&&mlp_gate.p&&native_down.p?PyTuple_Pack(4,prior.p,mlp_gate.p,native_down.p,indices):nullptr);
  Owned final(gate_fn.p&&final_args.p?PyObject_Call(gate_fn.p,final_args.p,dtype_kw.p):nullptr);
  return write("residual",final.p,5376,rows);
}
