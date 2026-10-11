// Offline observer of the UNMODIFIED checkpoint generate.py CLI.
// Capture its actual literal input/output lists, not re-decoded text or a new
// generator. This binary is never linked into LunaFlux's production runtime.
#include <Python.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

int install_upstream_stream_loader();
int install_upstream_prefill_partition();
int upstream_prefill_partition_probe();

struct Owned {
  PyObject *p;
  explicit Owned(PyObject *value) : p(value) {}
  ~Owned() { Py_XDECREF(p); }
  Owned(const Owned &) = delete;
};

// Explicit test-only capture at the first matched prefill frontier. Both
// rank directories are distinct; no forwarding arithmetic is replaced.
static int capture_prefill(PyObject *locals, PyObject *result) {
  const char *path = std::getenv("LUNA_UPSTREAM_LOGIT_CAPTURE");
  if (!path || !*path || !PyTuple_Check(result) || PyTuple_Size(result) != 3) return 0;
  auto *position = PyDict_GetItemString(locals, "start_pos");
  if (!position || !PyLong_Check(position)) return 0;
  const char *partition = std::getenv("LUNA_UPSTREAM_PREFILL_ROWS");
  if (partition && *partition) {
    auto *input=PyDict_GetItemString(locals,"input_ids");
    if(!input)return 0;
    Owned count(PyObject_CallMethod(input,"size","i",1));
    if(!count.p)return -1;
    // The wrapper supplies this invocation's full prompt extent. Capture
    // its final chunk, not the first chunk or a later decode invocation.
    const char *expected=std::getenv("LUNA_UPSTREAM_PREFILL_END");
    if(!expected)return 0;
    char *end=nullptr;long length=std::strtol(expected,&end,10);
    if(!end||*end||length<1||length>256){PyErr_SetString(PyExc_ValueError,"invalid prefill capture extent");return -1;}
    if(PyLong_AsLong(position)+PyLong_AsLong(count.p)!=length)return 0;
  } else if (PyLong_AsLong(position) != 0) return 0;
  auto *self = PyDict_GetItemString(locals, "self");
  if (!self) return 0;
  Owned cls(PyObject_GetAttrString(self, "__class__"));
  Owned name(cls.p ? PyObject_GetAttrString(cls.p, "__name__") : nullptr);
  if (!name.p) return -1;
  const char *class_name = PyUnicode_AsUTF8(name.p);
  if (!class_name) return -1;
  if (std::strcmp(class_name, "Transformer") != 0) return 0;
  Owned f32(PyObject_CallMethod(PyTuple_GetItem(result, 1), "float", nullptr));
  Owned cpu(f32.p ? PyObject_CallMethod(f32.p, "cpu", nullptr) : nullptr);
  Owned array(cpu.p ? PyObject_CallMethod(cpu.p, "numpy", nullptr) : nullptr);
  Owned bytes(array.p ? PyObject_CallMethod(array.p, "tobytes", nullptr) : nullptr);
  if (!bytes.p || !PyBytes_Check(bytes.p)) return -1;
  const auto size = PyBytes_Size(bytes.p);
  if (size != 129280 * 4) {
    PyErr_SetString(PyExc_ValueError, "original prefill vocabulary row has unexpected extent");
    return -1;
  }
  auto *file = std::fopen(path, "wbx");
  if (!file) { PyErr_SetFromErrnoWithFilename(PyExc_OSError, path); return -1; }
  const bool written = std::fwrite(PyBytes_AsString(bytes.p), 1, size, file) == static_cast<size_t>(size);
  const bool closed = std::fclose(file) == 0;
  if (!written || !closed) { PyErr_SetString(PyExc_OSError, "prefill capture write failed"); return -1; }
  std::printf("upstream_prefill_logits bytes=%zd numerical_path=unchanged\n", size);
  return 0;
}

// Inspect the same three pre-head layer means already retained by native
// DSpark. Only the matched prompt frontier is copied; original math is intact.
static int capture_layer(PyObject *locals, PyObject *result) {
  const char *path=std::getenv("LUNA_UPSTREAM_LOGIT_CAPTURE");
  if(!path||!*path||PyTuple_Check(result))return 0;
  auto *self=PyDict_GetItemString(locals,"self");
  auto *position=PyDict_GetItemString(locals,"start_pos");
  if(!self||!position||!PyLong_Check(position))return 0;
  Owned cls(PyObject_GetAttrString(self,"__class__"));
  Owned name(cls.p?PyObject_GetAttrString(cls.p,"__name__"):nullptr);
  const char *type=name.p?PyUnicode_AsUTF8(name.p):nullptr;
  if(!type)return -1;
  if(std::strcmp(type,"Block")!=0)return 0;
  Owned layer(PyObject_GetAttrString(self,"layer_id"));
  if(!layer.p)return -1;
  const long index=PyLong_AsLong(layer.p);
  if(PyErr_Occurred())return -1;
  if(index<40||index>42)return 0;
  Owned rows(PyObject_CallMethod(result,"size","i",1));
  if(!rows.p)return -1;
  const long count=PyLong_AsLong(rows.p);
  const char *partition=std::getenv("LUNA_UPSTREAM_PREFILL_ROWS");
  if(partition&&*partition) {
    const char *end=std::getenv("LUNA_UPSTREAM_PREFILL_END");
    if(!end||PyLong_AsLong(position)+count!=std::strtol(end,nullptr,10))return 0;
  } else if(PyLong_AsLong(position)!=0)return 0;
  Owned torch(PyImport_ImportModule("torch"));
  Owned bf16(torch.p?PyObject_GetAttrString(torch.p,"bfloat16"):nullptr);
  Owned u8(torch.p?PyObject_GetAttrString(torch.p,"uint8"):nullptr);
  Owned last(PyObject_CallMethod(result,"narrow","ili",1,count-1,1));
  Owned wide(last.p?PyObject_CallMethod(last.p,"float",nullptr):nullptr);
  Owned mean(wide.p?PyObject_CallMethod(wide.p,"mean","i",2):nullptr);
  Owned rounded(mean.p&&bf16.p?PyObject_CallMethod(mean.p,"to","O",bf16.p):nullptr);
  Owned cpu(rounded.p?PyObject_CallMethod(rounded.p,"cpu",nullptr):nullptr);
  Owned raw(cpu.p&&u8.p?PyObject_CallMethod(cpu.p,"view","O",u8.p):nullptr);
  Owned array(raw.p?PyObject_CallMethod(raw.p,"numpy",nullptr):nullptr);
  Owned bytes(array.p?PyObject_CallMethod(array.p,"tobytes",nullptr):nullptr);
  if(!bytes.p||PyBytes_Size(bytes.p)!=4096*2)return -1;
  const std::string filename=std::string(path)+".layer-"+std::to_string(index)+".bf16";
  auto *file=std::fopen(filename.c_str(),"wbx");
  if(!file){PyErr_SetFromErrnoWithFilename(PyExc_OSError,filename.c_str());return -1;}
  const bool written=std::fwrite(PyBytes_AsString(bytes.p),1,4096*2,file)==4096*2;
  const bool closed=std::fclose(file)==0;
  if(!written||!closed){PyErr_SetString(PyExc_OSError,"original layer mean capture failed");return -1;}
  std::printf("upstream_layer_mean layer=%ld start_pos=%ld rows=%ld frontier=last_literal_prompt_row\n",index,PyLong_AsLong(position),count);
  return 0;
}

static int emit(PyObject *input, PyObject *output) {
  if (!PyList_Check(input) || !PyList_Check(output) ||
      PyList_Size(input) != 1 || PyList_Size(output) != 1) {
    PyErr_SetString(PyExc_ValueError, "bounded upstream observer expects one sequence");
    return -1;
  }
  auto *prompt = PyList_GetItem(input, 0), *completion = PyList_GetItem(output, 0);
  if (!PyList_Check(prompt) || !PyList_Check(completion) ||
      PyList_Size(prompt) > 256 || PyList_Size(completion) > 9) {
    PyErr_SetString(PyExc_ValueError, "upstream observer extent exceeded");
    return -1;
  }
  Owned record(PyDict_New()), json(PyImport_ImportModule("json"));
  if (!record.p || !json.p ||
      PyDict_SetItemString(record.p, "prompt_tokens", input) < 0 ||
      PyDict_SetItemString(record.p, "completion_tokens", output) < 0) return -1;
  Owned encoded(PyObject_CallMethod(json.p, "dumps", "O", record.p));
  const char *text = encoded.p ? PyUnicode_AsUTF8(encoded.p) : nullptr;
  if (!text) return -1;
  std::printf("upstream_literal_tokens=%s\n", text);
  std::fflush(stdout);
  return 0;
}

// observe the actual sampler result, not generate.py's reporting-only EOS
// suffix. The caller's frontier distinguishes target and predictor calls.
static int emit_sample(PyFrameObject *frame, PyObject *result) {
  const char *path=std::getenv("LUNA_UPSTREAM_LOGIT_CAPTURE");
  if(!path||!*path)return 0;
  Owned caller(reinterpret_cast<PyObject *>(PyFrame_GetBack(frame)));
  Owned code(caller.p?reinterpret_cast<PyObject *>(PyFrame_GetCode(
    reinterpret_cast<PyFrameObject *>(caller.p))):nullptr);
  Owned name(code.p?PyObject_GetAttrString(code.p,"co_name"):nullptr);
  Owned locals(caller.p?PyFrame_GetLocals(reinterpret_cast<PyFrameObject *>(caller.p)):nullptr);
  if(!name.p||!locals.p)return -1;
  auto *position=PyDict_GetItemString(locals.p,"start_pos");
  if(!position||!PyLong_Check(position))return 0;
  Owned ids(PyList_Check(result)?Py_NewRef(result):PyObject_CallMethod(result,"tolist",nullptr));
  Owned json(PyImport_ImportModule("json"));
  Owned encoded(ids.p&&json.p?PyObject_CallMethod(json.p,"dumps","O",ids.p):nullptr);
  const char *text=encoded.p?PyUnicode_AsUTF8(encoded.p):nullptr;
  const char *function=PyUnicode_AsUTF8(name.p);
  if(!text||!function)return -1;
  std::printf("upstream_actual_sample caller=%s start_pos=%ld ids=%s reporting_suffix=false\n",
    function,PyLong_AsLong(position),text);
  std::fflush(stdout);
  return PyErr_Occurred()?-1:0;
}

static int trace(PyObject *filename, PyFrameObject *frame, int event, PyObject *result) {
  // A failing original function can emit a RETURN/None trace. Do not replace
  // its actual exception with an observer shape error.
  if (event != PyTrace_RETURN || result == Py_None) return 0;
  Owned code(reinterpret_cast<PyObject *>(PyFrame_GetCode(frame)));
  Owned file(code.p ? PyObject_GetAttrString(code.p, "co_filename") : nullptr);
  Owned name(code.p ? PyObject_GetAttrString(code.p, "co_name") : nullptr);
  if (!file.p || !name.p) return -1;
  const char *function = PyUnicode_AsUTF8(name.p);
  if (!function) return -1;
  if (std::strcmp(function,"sample")==0) return emit_sample(frame,result);
  if (std::strcmp(function, "forward") == 0) {
    Owned locals(PyFrame_GetLocals(frame));
    if(!locals.p)return -1;
    if(capture_layer(locals.p,result)<0)return -1;
    return capture_prefill(locals.p,result);
  }
  if (std::strcmp(function, "generate") != 0) return 0;
  const int matched = PyObject_RichCompareBool(file.p, filename, Py_EQ);
  if (matched <= 0) return matched;
  Owned locals(PyFrame_GetLocals(frame));
  if (!locals.p) return -1;
  auto *input = PyDict_GetItemString(locals.p, "prompt_tokens");
  if (!input) { PyErr_SetString(PyExc_ValueError, "original generation input missing"); return -1; }
  return emit(input, result);
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fputs("ORIGINAL_GENERATE_SCRIPT [ORIGINAL_CLI_ARGS] | --probe\n", stderr); return 2; }
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.parse_argv = 0;
  auto status = PyConfig_SetBytesArgv(&config, argc - 1, argv + 1);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) { std::fputs("upstream observer interpreter initialization failed\n", stderr); return 1; }
  int failed = 0;
  {
    if (std::strcmp(argv[1], "--partition-probe") == 0) {
      failed=upstream_prefill_partition_probe();
    } else if (std::strcmp(argv[1], "--probe") == 0) {
      Owned input(Py_BuildValue("[[ii]]", 5, 223)), output(Py_BuildValue("[[ii]]", 19923, 1));
      failed = emit(input.p, output.p);
    } else {
      if(install_upstream_stream_loader()<0) failed=-1;
      std::string path(argv[1]);
      const auto slash = path.find_last_of('/');
      Owned directory(PyUnicode_FromString(path.substr(0, slash).c_str()));
      Owned filename(PyUnicode_FromString(path.c_str()));
      auto *paths = PySys_GetObject("path");
      if (!directory.p || !filename.p || !paths || PyList_Insert(paths, 0, directory.p) < 0) failed = -1;
      if(!failed && install_upstream_prefill_partition()<0) failed=-1;
      auto *file = failed ? nullptr : std::fopen(path.c_str(), "r");
      if (!file) { std::fputs("original CLI cannot be opened\n", stderr); failed = -1; }
      else {
        PyEval_SetTrace(trace, filename.p);
        failed = PyRun_SimpleFileExFlags(file, path.c_str(), 1, nullptr);
        PyEval_SetTrace(nullptr, nullptr);
      }
    }
    if (PyErr_Occurred()) { PyErr_Print(); failed = -1; }
  }
  if (Py_FinalizeEx() < 0) failed = -1;
  return failed ? 1 : 0;
}
