#pragma once
#include "original_tensor_io.h"

// Offline checkpoint layout only; math stays in the original module.
inline bool h3_parameter(PyObject *object, PyObject *open, PyObject *map,
                         const char *root, const char *target, const char *source,
                         const char *part=nullptr) {
  Owned destination(PyObject_GetAttrString(object,target));
  Owned value(weight(open,map,root,source));
  if(!destination.p||!value.p)return false;
  if(part) {
    Owned loader(PyObject_GetAttrString(destination.p,"weight_loader"));
    Owned args(Py_BuildValue("OOs",destination.p,value.p,part));
    Owned copied(loader.p?PyObject_CallObject(loader.p,args.p):nullptr);
    return copied.p!=nullptr;
  }
  Owned copied(PyObject_CallMethod(destination.p,"copy_","O",value.p));
  return copied.p!=nullptr;
}

inline bool h3_attention_mlp(PyObject *block,PyObject *open,PyObject *map,
                             const char *root,const std::string &prefix) {
  Owned attention(PyObject_GetAttrString(block,"attn"));
  Owned qkv(attention.p?PyObject_GetAttrString(attention.p,"qkv_proj"):nullptr);
  if(!qkv.p)return false;
  for(const char *part:{"q","k","v"}) {
    const std::string name=prefix+"attn.to_"+std::string(part)+".weight";
    if(!h3_parameter(qkv.p,open,map,root,"weight",name.c_str(),part))return false;
  }
  for(const auto &pair:std::vector<std::pair<const char*,const char*>>{
      {"q_norm","attn.norm_q.weight"},{"k_norm","attn.norm_k.weight"},
      {"out_proj","attn.to_out.0.weight"}}) {
    Owned child(PyObject_GetAttrString(attention.p,pair.first));
    if(!child.p||!h3_parameter(child.p,open,map,root,"weight",(prefix+pair.second).c_str()))return false;
  }
  for(const char *name:{"norm1","norm2"}) {
    Owned child(PyObject_GetAttrString(block,name));
    if(!child.p||!h3_parameter(child.p,open,map,root,"weight",(prefix+name+".weight").c_str()))return false;
  }
  Owned mlp(PyObject_GetAttrString(block,"mlp"));
  Owned fc1(mlp.p?PyObject_GetAttrString(mlp.p,"fc1"):nullptr);
  Owned fc2(mlp.p?PyObject_GetAttrString(mlp.p,"fc2"):nullptr);
  Owned target(fc1.p?PyObject_GetAttrString(fc1.p,"weight"):nullptr);
  Owned loader(target.p?PyObject_GetAttrString(target.p,"weight_loader"):nullptr);
  Owned fused(weight(open,map,root,(prefix+"ff.net.0.proj.weight").c_str()));
  if(!loader.p||!fused.p||!fc2.p)return false;
  // Converted [up,gate] -> original [gate,up].
  for(int shard=0;shard<2;++shard) {
    Owned half(PyObject_CallMethod(fused.p,"narrow","iii",0,(1-shard)*14336,14336));
    Owned args(half.p?Py_BuildValue("OOi",target.p,half.p,shard):nullptr);
    Owned copied(args.p?PyObject_CallObject(loader.p,args.p):nullptr);
    if(!copied.p)return false;
  }
  return h3_parameter(fc2.p,open,map,root,"weight",(prefix+"ff.net.2.weight").c_str());
}
