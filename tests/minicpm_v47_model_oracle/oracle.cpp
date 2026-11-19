// Offline numerical adapter. All model arithmetic executes the unmodified
// Transformers 5.18.0 / PyTorch implementation. No production runtime linkage.
#include <Python.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

struct Object {
  PyObject* p = nullptr;
  explicit Object(PyObject* value) : p(value) {
    if (!p) { PyErr_Print(); throw std::runtime_error("upstream model call failed"); }
  }
  ~Object() { Py_XDECREF(p); }
  Object(const Object&) = delete;
  Object& operator=(const Object&) = delete;
  Object(Object&& other) noexcept : p(std::exchange(other.p, nullptr)) {}
  Object& operator=(Object&& other) noexcept {
    if (this != &other) { Py_XDECREF(p); p = std::exchange(other.p, nullptr); }
    return *this;
  }
};

static Object attr(PyObject* object, const char* name) {
  return Object(PyObject_GetAttrString(object, name));
}
static Object text(const std::string& value) {
  return Object(PyUnicode_FromStringAndSize(value.data(), value.size()));
}
static Object integer(long value) { return Object(PyLong_FromLong(value)); }
static void put(PyObject* object, const char* name, PyObject* value) {
  if (PyDict_SetItemString(object, name, value) < 0) throw std::runtime_error("reference dictionary write failed");
}
static void append(PyObject* array, PyObject* value) {
  if (PyList_Append(array, value) < 0) throw std::runtime_error("reference list write failed");
}
static Object call(PyObject* object, const char* name, PyObject* args = nullptr, PyObject* kwargs = nullptr) {
  Object function = attr(object, name);
  Object empty(PyTuple_New(0));
  return Object(PyObject_Call(function.p, args ? args : empty.p, kwargs));
}
static Object load_json(PyObject* json, const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open reference input: " + path);
  const std::string source((std::istreambuf_iterator<char>(input)), {});
  Object source_text = text(source);
  Object args(PyTuple_Pack(1, source_text.p));
  return call(json, "loads", args.p);
}
static PyObject* field(PyObject* object, const char* name) {
  PyObject* value = PyDict_GetItemString(object, name);
  if (!value) throw std::runtime_error(std::string("missing reference field: ") + name);
  return value;
}
static long number(PyObject* value) {
  const long result = PyLong_AsLong(value);
  if (PyErr_Occurred()) { PyErr_Print(); throw std::runtime_error("expected reference integer"); }
  return result;
}

static Object tensor(PyObject* torch, PyObject* values) {
  Object rows(PyList_New(0));
  append(rows.p, values);
  Object args(PyTuple_Pack(1, rows.p));
  Object kwargs(PyDict_New());
  Object dtype = attr(torch, "long");
  Object device = text("cuda:0");
  put(kwargs.p, "dtype", dtype.p);
  put(kwargs.p, "device", device.p);
  return call(torch, "tensor", args.p, kwargs.p);
}
static Object forward(PyObject* model, PyObject* torch, PyObject* ids, PyObject* cache, long total) {
  Object input = tensor(torch, ids);
  Object one = integer(1), width = integer(total);
  Object shape(PyTuple_Pack(2, one.p, width.p));
  Object mask_args(PyTuple_Pack(1, shape.p));
  Object mask_kwargs(PyDict_New());
  Object dtype = attr(torch, "long"), device = text("cuda:0");
  put(mask_kwargs.p, "dtype", dtype.p);
  put(mask_kwargs.p, "device", device.p);
  Object mask = call(torch, "ones", mask_args.p, mask_kwargs.p);
  Object kwargs(PyDict_New());
  put(kwargs.p, "input_ids", input.p);
  put(kwargs.p, "attention_mask", mask.p);
  put(kwargs.p, "past_key_values", cache);
  put(kwargs.p, "use_cache", Py_True);
  put(kwargs.p, "logits_to_keep", one.p);
  Object args(PyTuple_New(0));
  return Object(PyObject_Call(model, args.p, kwargs.p));
}

static Object run_request(PyObject* model, PyObject* torch, PyObject* request, const std::string& root, long index) {
  PyObject* ids = field(request, "token_ids");
  const Py_ssize_t count = PyList_Size(ids);
  const long maximum = number(field(request, "maximum_new_tokens"));
  if (count <= 0 || count > 4096 || maximum <= 0 || maximum > 128) {
    throw std::runtime_error("reference request outside initial text profile");
  }
  Object inner = attr(model, "model");
  if (PyObject_SetAttrString(inner.p, "rope_deltas", Py_None) < 0) throw std::runtime_error("cannot reset upstream rope state");
  Object cache(Py_NewRef(Py_None)), output(Py_NewRef(Py_None));
  // Use the native deployment's 32-row prefill boundaries. The original
  // upstream cache and recurrent/full-attention implementations remain intact.
  for (Py_ssize_t cursor = 0; cursor < count;) {
    const Py_ssize_t end = std::min(count, cursor + 32);
    Object chunk(PyList_GetSlice(ids, cursor, end));
    output = forward(model, torch, chunk.p, cache.p, end);
    cache = attr(output.p, "past_key_values");
    cursor = end;
  }
  Object sampled(PyList_New(0)), emitted(PyList_New(0)), paths(PyList_New(0));
  bool stopped = false;
  for (long step = 0; step < maximum; ++step) {
    Object logits = attr(output.p, "logits");
    Object minus_one = integer(-1), reshape_args(PyTuple_Pack(1, minus_one.p));
    Object flat = call(logits.p, "reshape", reshape_args.p);
    Object fp32 = call(flat.p, "float");
    Object cpu = call(fp32.p, "cpu");
    Object contiguous = call(cpu.p, "contiguous");
    Object finite_args(PyTuple_Pack(1, contiguous.p));
    Object finite = call(torch, "isfinite", finite_args.p);
    Object all = call(finite.p, "all"), valid = call(all.p, "item");
    if (PyObject_IsTrue(valid.p) != 1) throw std::runtime_error("upstream logits are not finite");
    Object array = call(contiguous.p, "numpy");
    const std::string path = root + "/request-" + std::to_string(index) + "-step-" + std::to_string(step) + ".f32";
    Object path_text = text(path), file_args(PyTuple_Pack(1, path_text.p));
    call(array.p, "tofile", file_args.p);
    append(paths.p, path_text.p);
    Object argmax = call(contiguous.p, "argmax"), next = call(argmax.p, "item");
    const long token = number(next.p);
    append(sampled.p, next.p);
    // Exact stop IDs from the pinned generation_config.json and native profile.
    if (token == 248044 || token == 248046) { stopped = true; break; }
    append(emitted.p, next.p);
    if (step + 1 != maximum) {
      Object next_ids(PyList_New(0));
      append(next_ids.p, next.p);
      output = forward(model, torch, next_ids.p, cache.p, count + step + 1);
      cache = attr(output.p, "past_key_values");
    }
  }
  Object report(PyDict_New());
  put(report.p, "id", field(request, "id"));
  put(report.p, "input_token_ids", ids);
  put(report.p, "sampled_token_ids", sampled.p);
  put(report.p, "emitted_token_ids", emitted.p);
  put(report.p, "logits_files", paths.p);
  Object reason = text(stopped ? "stop" : "length");
  put(report.p, "finish_reason", reason.p);
  return report;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: minicpm-model-oracle MODEL_ROOT NATIVE_PLAN.json NEW_OUTPUT\n";
    return 2;
  }
  try {
    if (!std::filesystem::create_directory(argv[3])) throw std::runtime_error("reference output must be new");
    Py_Initialize();
    Object json(PyImport_ImportModule("json")), torch(PyImport_ImportModule("torch"));
    Object transformers(PyImport_ImportModule("transformers"));
    Object version = attr(transformers.p, "__version__");
    const char* version_text = PyUnicode_AsUTF8(version.p);
    if (!version_text || std::string(version_text) != "5.18.0") throw std::runtime_error("reference requires Transformers 5.18.0");
    Object cuda = attr(torch.p, "cuda"), available = call(cuda.p, "is_available");
    if (PyObject_IsTrue(available.p) != 1) throw std::runtime_error("reference requires real CUDA");
    Object disabled(PyTuple_Pack(1, Py_False));
    call(torch.p, "set_grad_enabled", disabled.p);
    Object highest = text("highest"), precision_args(PyTuple_Pack(1, highest.p));
    call(torch.p, "set_float32_matmul_precision", precision_args.p);
    Object model_class = attr(transformers.p, "AutoModelForImageTextToText");
    Object path = text(argv[1]), args(PyTuple_Pack(1, path.p)), kwargs(PyDict_New());
    Object bf16 = attr(torch.p, "bfloat16"), device = text("cpu"), eager = text("eager");
    put(kwargs.p, "dtype", bf16.p);
    put(kwargs.p, "device_map", device.p);
    put(kwargs.p, "attn_implementation", eager.p);
    put(kwargs.p, "local_files_only", Py_True);
    put(kwargs.p, "use_safetensors", Py_True);
    put(kwargs.p, "disable_mmap", Py_False);
    Object model = call(model_class.p, "from_pretrained", args.p, kwargs.p);
    Object footprint = call(model.p, "get_memory_footprint");
    const long payload = number(footprint.p);
    if (payload <= 0 || payload > 80000000000L) throw std::runtime_error("unexpected upstream model payload");
    std::cout << "CPU weight load complete; model payload " << payload << " bytes; moving to CUDA\n" << std::flush;
    Object cuda_device = text("cuda:0"), move_args(PyTuple_Pack(1, cuda_device.p));
    call(model.p, "to", move_args.p);
    Object parameters = call(model.p, "parameters");
    Object iterator(PyObject_GetIter(parameters.p));
    while (PyObject* item = PyIter_Next(iterator.p)) {
      Object parameter(item), location = attr(parameter.p, "device"), type = attr(location.p, "type");
      const char* name = PyUnicode_AsUTF8(type.p);
      if (!name || std::string(name) != "cuda") throw std::runtime_error("upstream parameter did not move to CUDA");
    }
    if (PyErr_Occurred()) throw std::runtime_error("cannot inspect upstream parameter devices");
    std::cout << "all model parameters are on CUDA\n" << std::flush;
    call(model.p, "eval");
    Object plan = load_json(json.p, argv[2]);
    PyObject* requests = field(plan.p, "requests");
    const Py_ssize_t count = PyList_Size(requests);
    if (count <= 0 || count > 16) throw std::runtime_error("reference expects 1..16 text requests");
    Object rows(PyList_New(0));
    for (Py_ssize_t i = 0; i < count; ++i) {
      Object result = run_request(model.p, torch.p, PyList_GetItem(requests, i), argv[3], i);
      append(rows.p, result.p);
    }
    Object report(PyDict_New()), format = text("lunaflux.minicpm-v47-upstream-model.v1");
    Object torch_version = attr(torch.p, "__version__");
    Object mode = text("unchanged upstream eager model; 32-token cached prefill; greedy decode");
    put(report.p, "format", format.p);
    put(report.p, "transformers_version", version.p);
    put(report.p, "torch_version", torch_version.p);
    put(report.p, "reference_mode", mode.p);
    put(report.p, "requests", rows.p);
    put(report.p, "gpu_execution", Py_True);
    put(report.p, "model_payload_bytes", footprint.p);
    Object peak_allocated = call(cuda.p, "max_memory_allocated");
    Object peak_reserved = call(cuda.p, "max_memory_reserved");
    put(report.p, "cuda_peak_allocated_bytes", peak_allocated.p);
    put(report.p, "cuda_peak_reserved_bytes", peak_reserved.p);
    Object dump_args(PyTuple_Pack(1, report.p));
    Object serialized = call(json.p, "dumps", dump_args.p);
    const char* serialized_text = PyUnicode_AsUTF8(serialized.p);
    if (!serialized_text) throw std::runtime_error("cannot serialize reference result");
    std::ofstream file(std::string(argv[3]) + "/report.json");
    file << serialized_text << '\n';
    if (!file) throw std::runtime_error("cannot write reference report");
    std::cout << "upstream MiniCPM model reference complete\n";
    return 0;
  } catch (const std::exception& error) {
    if (Py_IsInitialized() && PyErr_Occurred()) PyErr_Print();
    std::cerr << error.what() << '\n';
    return 1;
  }
}
