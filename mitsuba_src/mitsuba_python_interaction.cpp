#include "mitsuba_python_interaction.h"
#include "LiteMath.h"
#include <chrono>

//some python libraries (e.g. numpy) requires python context
//to be initialized only once. It is a know issue and we
//can do nothing about it.
bool ever_initialized = false;

#define DEL(X) if (X) {Py_DECREF(X);}
void MitsubaInterface::show_errors()
{
  PyObject *pExcType, *pExcValue, *pExcTraceback;
  PyErr_Fetch(&pExcType, &pExcValue, &pExcTraceback);
  if (pExcType != NULL)
  {
    PyObject *pRepr = PyObject_Repr(pExcType);
    logerr("An error occurred:");
    logerr("- EXC type: %s", PyUnicode_AsUTF8(pRepr));
    Py_DecRef(pRepr);
    Py_DecRef(pExcType);
  }
  if (pExcValue != NULL)
  {
    PyObject *pRepr = PyObject_Repr(pExcValue);
    logerr("An error occurred:");
    logerr("- EXC value: %s", PyUnicode_AsUTF8(pRepr));
    Py_DecRef(pRepr);
    Py_DecRef(pExcValue);
  }
  if (pExcTraceback != NULL)
  {
    PyObject *pRepr = PyObject_Repr(pExcTraceback);
    logerr("An error occurred:");
    logerr("- EXC traceback: %s", PyUnicode_AsUTF8(pRepr));
    Py_DecRef(pRepr);
    Py_DecRef(pExcTraceback);
  }
}

void MitsubaInterface::finish()
{
  for (int i = 0; i < buffers.size(); i++)
  {
    if (buffers[i])
    {
      delete[] buffers[i];
      buffers[i] = nullptr;
    }
  }
  buffers.clear();
  buffer_names.clear();
  model_max_size = 0;
}

MitsubaInterface::~MitsubaInterface()
{
  DEL(mitsubaContext);
  DEL(pModule);
  Py_Finalize();
}

MitsubaInterface::MitsubaInterface(const std::string &scripts_dir, const std::string &file_name)
{
  //Interpreter initialization
  std::string append_path_str = std::string("sys.path.append(\"")+scripts_dir+"\")";
  if (!ever_initialized)
  {
    Py_Initialize();
    ever_initialized = true;
  }
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("import os");
  PyRun_SimpleString(append_path_str.c_str());
  PyObject *pName;
  pName = PyUnicode_FromString(file_name.c_str());
  pModule = PyImport_Import(pName);
  DEL(pName);
  if (!pModule)
    show_errors();
}

void MitsubaInterface::init_scene_and_settings(RenderSettings _render_settings, ModelInfo _model_info)
{
  finish();
  render_settings = _render_settings;
  model_info = _model_info;

  int mesh_parts_count = model_info.parts.size();
  for (int part_id = 0; part_id < mesh_parts_count; part_id++)
  {
    buffer_names.push_back("vertex_positions_"+std::to_string(part_id));
    buffers.push_back(nullptr);

    buffer_names.push_back("vertex_normals_"+std::to_string(part_id));
    buffers.push_back(nullptr);

    buffer_names.push_back("vertex_texcoords_"+std::to_string(part_id));
    buffers.push_back(nullptr);
  }
  buffer_names.push_back("camera_params");
  buffers.push_back(nullptr);

  render_settings = _render_settings;
  //mitsuba context initialization
  std::string mitsuba_var = "";
  switch (render_settings.mitsubaVar)
  {
  case MitsubaVariant::CUDA :
    mitsuba_var = "cuda_ad_rgb";
    break;
  case MitsubaVariant::LLVM :
    mitsuba_var = "llvm_ad_rgb";
    break;
  default:
    mitsuba_var = "cuda_ad_rgb";
    break;
  }

  std::string render_style = "";
  switch (render_settings.renderStyle)
  {
  case RenderStyle::SILHOUETTE:
    render_style = "silhouette";
    break;
  case RenderStyle::MONOCHROME:
    render_style = "monochrome";
    break;
  case RenderStyle::TEXTURED_CONST:
    render_style = "textured_const";
    break;
  case RenderStyle::TEXTURED_DEMO:
    render_style = "textured_demo";
    break;
  case RenderStyle::MONOCHROME_DEMO:
    render_style = "monochrome_demo";
    break;
  default:
    render_style = "silhouette";
    break;
  }

  std::string texture_names = "";
  std::string material_names = "";
  for (int i=0;i<model_info.parts.size();i++)
  {
    if (i != 0)
    {
      texture_names +="|";
      material_names +="|";
    }
    texture_names += model_info.parts[i].texture_name;
    material_names += model_info.parts[i].material_name;
  }

  PyObject *initFunc, *initArgs, *basePath, *iw_arg, *ih_arg, *spp_arg, *mv, *rs, *tn, *mn;
  basePath = PyUnicode_FromString("data/");
  iw_arg = PyLong_FromLong(render_settings.image_w);
  ih_arg = PyLong_FromLong(render_settings.image_h);
  spp_arg = PyLong_FromLong(render_settings.samples_per_pixel);
  mv = PyUnicode_FromString(mitsuba_var.c_str());
  rs = PyUnicode_FromString(render_style.c_str());
  tn = PyUnicode_FromString(texture_names.c_str());
  mn = PyUnicode_FromString(material_names.c_str());
  initArgs = PyTuple_Pack(8, basePath, iw_arg, ih_arg, spp_arg, mv, rs, tn, mn);
  
  initFunc = PyObject_GetAttrString(pModule, (char *)"init");
  if (!initFunc)
    show_errors();
  
  if (mitsubaContext)
    DEL(mitsubaContext);

  mitsubaContext = PyObject_CallObject(initFunc, initArgs);
  if (!mitsubaContext)
    show_errors();
  
  DEL(initFunc);
  DEL(initArgs);
  DEL(basePath);
  DEL(iw_arg);
  DEL(ih_arg);
  DEL(spp_arg);
  DEL(mv);
  DEL(rs);
  DEL(tn);
  DEL(mn);
}

std::string get_loss_function_name(MitsubaInterface::LossFunction loss_function)
{
  std::string loss_function_name = "F_loss_mse";
  switch (loss_function)
  {
  case MitsubaInterface::LossFunction::LOSS_MSE :
    loss_function_name = "F_loss_mse";
    break;

  case MitsubaInterface::LossFunction::LOSS_MSE_SQRT :
    loss_function_name = "F_loss_mse_sqrt";
    break;
  
  case MitsubaInterface::LossFunction::LOSS_MIXED :
    loss_function_name = "F_loss_mixed";
    break;

  default:
    loss_function_name = "F_loss_mse";
    break;
  }
  return loss_function_name;
}
void MitsubaInterface::init_optimization(const std::vector<std::string> &reference_image_dir, LossFunction loss_function,
                                         RenderSettings render_settings, ModelInfo model_info,
                                         bool save_intermediate_images)
{
  init_optimization_internal("init_optimization", reference_image_dir, loss_function, render_settings, model_info,
                             0, save_intermediate_images);
}

void MitsubaInterface::init_optimization_internal(const std::string &function_name, const std::vector<std::string> &reference_images_dir,
                                                  LossFunction loss_function, RenderSettings render_settings, ModelInfo model_info,
                                                  float texture_rec_learing_rate, bool save_intermediate_images)
{
  init_scene_and_settings(render_settings, model_info);

  std::string loss_function_name = get_loss_function_name(loss_function);
  
  //save all strings as "ref1.png#ref2.png#ref3.png"
  std::string full_ref_string = "";
  for (int i=0;i<reference_images_dir.size();i++)
  {
    full_ref_string += reference_images_dir[i];
    if (i < reference_images_dir.size() - 1)
     full_ref_string += "#"; 
  }

  PyObject *func, *args, *ref_dir_arg, *func_ret, *loss_func, *lr, *c_cnt, *int_im;

  func = PyObject_GetAttrString(pModule, function_name.c_str());
  ref_dir_arg = PyUnicode_FromString(full_ref_string.c_str());
  loss_func = PyObject_GetAttrString(pModule, loss_function_name.c_str());
  lr = PyFloat_FromDouble(texture_rec_learing_rate);
  int_im = PyLong_FromLong((int)save_intermediate_images);
  c_cnt = PyLong_FromLong(reference_images_dir.size());
  args = PyTuple_Pack(6, mitsubaContext, ref_dir_arg, loss_func, lr, c_cnt, int_im);
  func_ret = PyObject_CallObject(func, args);
  show_errors();

  iteration = 0;

  DEL(func);
  DEL(args);
  DEL(ref_dir_arg);
  DEL(func_ret);
  DEL(loss_func);
  DEL(lr);
  DEL(c_cnt);
  DEL(int_im);
}

void MitsubaInterface::init_optimization_with_tex(const std::vector<std::string> &reference_image_dir, LossFunction loss_function, 
                                                  RenderSettings render_settings, ModelInfo model_info, 
                                                  float texture_rec_learing_rate,
                                                  bool save_intermediate_images)
{
  render_settings.renderStyle = RenderStyle::TEXTURED_CONST;
  init_optimization_internal("init_optimization_with_tex", reference_image_dir, loss_function, render_settings,
                             model_info, texture_rec_learing_rate, save_intermediate_images);
}

void MitsubaInterface::model_to_ctx(const DFModel &model)
{
  active_parts.clear();
  int start_buffer_offset = 0;
  PartOffsets off = model.second;
  off.push_back({"", model.first.size()});//to make offset calculations simplier

  for (auto &part : model_info.parts)
  {
    int part_size = 0;
    const float *part_data = nullptr;
    for (int i=0;i<off.size()-1;i++)
    {
      if (off[i].first == part.name)
      {
        part_data = model.first.data() + off[i].second;
        part_size = off[i+1].second - off[i].second;
      }
    }

    float placeholder_triangle[] = 
    {
      0,0,0, 0,0,1, 0,0,
      0.001,0,0, 0,0,1, 0,1,
      0,0.001,0, 0,0,1, 1,0
    };
    if (part_data == nullptr || part_size <= 0)
    {
      //This part does not exist in model (it's OK) or corrupted
      part_data = placeholder_triangle;
      part_size = sizeof(placeholder_triangle)/sizeof(float);
    }
    else
    {
      active_parts.push_back(start_buffer_offset/3);
    }

    auto &ml = model_info.layout;
    int vertex_count = part_size / ml.f_per_vert;
    if (model_max_size < vertex_count)
      set_model_max_size(vertex_count);
    assert(start_buffer_offset + ml.offsets.size() - 1 <= buffers.size());
    for (int i=0;i<ml.offsets.size() - 1;i++)
    {
      int offset = ml.offsets[i];
      int size = ml.offsets[i + 1] - ml.offsets[i];
      if (offset >= 0 && size > 0)
      {
        int b_id = start_buffer_offset + i;
        clear_buffer(b_id, 0.0f);
        for (int j = 0; j < vertex_count; j++)
          memcpy(buffers[b_id] + size*j, part_data + ml.f_per_vert * j + offset, sizeof(float)*size);
        set_array_to_ctx_internal(buffer_names[b_id], b_id, size * vertex_count);
      }
    }
    show_errors();
    start_buffer_offset += 3;
  }
}

void MitsubaInterface::camera_to_ctx(const CamInfo &camera, std::string camera_name)
{
  PyObject *func, *args, *p_name, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9, *p10, *p11, *p12, *func_ret;

  func = PyObject_GetAttrString(pModule, (char *)"set_camera");

  p_name = PyUnicode_FromString(camera_name.c_str());

  p1 = PyFloat_FromDouble(camera.origin.x);
  p2 = PyFloat_FromDouble(camera.origin.y);
  p3 = PyFloat_FromDouble(camera.origin.z);

  p4 = PyFloat_FromDouble(camera.target.x);
  p5 = PyFloat_FromDouble(camera.target.y);
  p6 = PyFloat_FromDouble(camera.target.z);

  p7 = PyFloat_FromDouble(camera.up.x);
  p8 = PyFloat_FromDouble(camera.up.y);
  p9 = PyFloat_FromDouble(camera.up.z);

  p10 = PyFloat_FromDouble(180 * camera.fov_rad / LiteMath::M_PI);
  p11 = PyLong_FromLong(render_settings.image_w);
  p12 = PyLong_FromLong(render_settings.image_h);

  args = PyTuple_Pack(14, mitsubaContext, p_name, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
  func_ret = PyObject_CallObject(func, args);
  show_errors();

  DEL(func);
  DEL(args);
  DEL(p_name);
  DEL(p1);
  DEL(p2);
  DEL(p3);
  DEL(p4);
  DEL(p5);
  DEL(p6);
  DEL(p7);
  DEL(p8);
  DEL(p9);
  DEL(p10);
  DEL(p11);
  DEL(p12);
  DEL(func_ret);
}

void MitsubaInterface::render_model_to_file(const DFModel &model, const std::string &image_dir,
                                            const CamInfo &camera, const std::vector<float> &scene_params)
{  
  model_to_ctx(model);
  camera_to_ctx(camera, "camera");

  int cameras_buf_id = get_camera_buffer_id();
  assert(scene_params.size() > 0);
  memcpy(buffers[cameras_buf_id], scene_params.data(), sizeof(float)*scene_params.size());
  set_array_to_ctx_internal(buffer_names[cameras_buf_id], cameras_buf_id, scene_params.size());
  show_errors();

  PyObject *func, *args, *ref_dir_arg, *func_ret;

  func = PyObject_GetAttrString(pModule, (char *)"render_and_save_to_file");
  ref_dir_arg = PyUnicode_FromString(image_dir.c_str());
  args = PyTuple_Pack(2, mitsubaContext, ref_dir_arg);
  func_ret = PyObject_CallObject(func, args);
  show_errors();

  DEL(func);
  DEL(args);
  DEL(ref_dir_arg);
  DEL(func_ret);
}

float MitsubaInterface::render_and_compare(const DFModel &model, const std::vector<CamInfo> &cameras, const std::vector<float> &scene_params,
                                           double *timers)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  model_to_ctx(model);
  for (int i=0;i<cameras.size();i++)
    camera_to_ctx(cameras[i], "camera_"+std::to_string(i));

  int cameras_buf_id = get_camera_buffer_id();
  assert(scene_params.size() > 0);
  memcpy(buffers[cameras_buf_id], scene_params.data(), sizeof(float)*scene_params.size());
  set_array_to_ctx_internal(buffer_names[cameras_buf_id], cameras_buf_id, scene_params.size());
  show_errors();

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  float loss = render_and_compare_internal();
  std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
  //get derivatives by vertex positions and scene parameters
  get_array_from_ctx_internal(buffer_names[0] + "_grad", 0);
  get_array_from_ctx_internal(buffer_names[cameras_buf_id] + "_grad", cameras_buf_id);
  std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
  if (timers)
  {
    timers[2] += 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    timers[3] += 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    timers[4] += 1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
  }
  return loss;
}

void MitsubaInterface::compute_final_grad(const std::vector<float> &jac, int params_count, int vertex_count, 
                                          std::vector<float> &final_grad)
{
  auto &ml = model_info.layout;
  for (int part_id : active_parts)
  {
    // offsets[0] offset always represent positions. We do not calculate derivatives by other channels (normals, tc)
    int offset = ml.offsets[0];
    int size = ml.offsets[1] - ml.offsets[0];
    if (offset >= 0 && size > 0)
    {
      for (int i = 0; i < vertex_count; i++)
      {
        for (int j = 0; j < params_count; j++)
        {
          for (int k = 0; k < size; k++)
          {
            final_grad[j] += jac[(ml.f_per_vert * i + offset + k) * params_count + j] * buffers[3*part_id][size * i + k];
          }
        }
      }
    }
  }

  for (int i = params_count; i < final_grad.size(); i++)
    final_grad[i] += buffers[get_camera_buffer_id()][i - params_count]; // gradient by camera params
}

void MitsubaInterface::get_pos_derivatives(float *out_grad, int vertex_count)
{
  int off = 0; 
  auto &ml = model_info.layout;
  assert(active_parts.size()==1);
  for (int part_id : active_parts)
  {
    // offsets[0] offset always represent positions. We do not calculate derivatives by other channels (normals, tc)
    int size = ml.offsets[1] - ml.offsets[0];
    assert(size == 3);
    for (int i = 0; i < vertex_count; i++)
    {
      for (int k = 0; k < size; k++)
        out_grad[off + size * i + k] = buffers[3*part_id][size * i + k];
    }
    off += vertex_count*size;
  }
}

void MitsubaInterface::set_model_max_size(int _model_max_size)
{
  _model_max_size = std::max(64, _model_max_size);
  model_max_size = _model_max_size;
  if (model_max_size >= 0)
  {
    for (int i=0;i<buffers.size(); i++)
    {
      if (buffers[i])
        delete[] buffers[i];
      buffers[i] = new float[4*model_max_size];
    }
  }
}

int MitsubaInterface::get_array_from_ctx_internal(const std::string &name, int buffer_id)
{
  PyObject *func, *args, *params, *params_bytes, *params_name;
  params_name = PyUnicode_FromString(name.c_str());
  args = PyTuple_Pack(2, mitsubaContext, params_name);
  func = PyObject_GetAttrString(pModule, (char *)"get_params");
  params = PyObject_CallObject(func, args);
  if (!params)
    show_errors();
  params_bytes = PyObject_Bytes(params);
  if (!params_bytes)
    show_errors();

  int sz = PyBytes_Size(params_bytes);
  int data_floats = sz / sizeof(float);
  if (data_floats > 4*model_max_size)
  {
    logerr("Python array %s contains %d float, while buffer size is %d. Some data will be ignored", name.c_str(), data_floats, 4*model_max_size);
  }
  char *data = PyBytes_AsString(params_bytes);
  memcpy(buffers[buffer_id], data, std::min<int>(sz, 4*model_max_size*sizeof(float)));
  DEL(args);
  DEL(func);
  DEL(params);
  DEL(params_bytes);
  DEL(params_name);

  return data_floats;
}

void MitsubaInterface::set_array_to_ctx_internal(const std::string &name, int buffer_id, int size)
{
  PyObject *func, *args, *params_n, *params_bytes, *params, *params_name;
  params_name = PyUnicode_FromString(name.c_str());
  params_n = PyLong_FromLong(size);
  params_bytes = PyBytes_FromStringAndSize((const char *)buffers[buffer_id], sizeof(float) * size);
  args = PyTuple_Pack(4, mitsubaContext, params_name, params_bytes, params_n);
  func = PyObject_GetAttrString(pModule, (char *)"set_params");
  params = PyObject_CallObject(func, args);

  DEL(args);
  DEL(func);
  DEL(params);
  DEL(params_n);
  DEL(params_bytes);
  DEL(params_name);
}

float MitsubaInterface::render_and_compare_internal()
{
  PyObject *pFunc, *pIndex, *pArgs, *pValue;

  pFunc = PyObject_GetAttrString(pModule, (char *)"render");
  if (!pFunc)
    show_errors();
  pIndex = PyLong_FromLong(iteration);
  iteration++;
  pArgs = PyTuple_Pack(2, pIndex, mitsubaContext);
  pValue = PyObject_CallObject(pFunc, pArgs);
  if (!pValue)
    show_errors();
  double result = PyFloat_AsDouble(pValue);

  DEL(pValue);
  DEL(pIndex);
  DEL(pArgs);
  DEL(pFunc);

  return result;
}

void MitsubaInterface::clear_buffer(int buffer_id, float val)
{
  std::fill_n(buffers[buffer_id], model_max_size, val);
}

std::vector<std::string> MitsubaInterface::get_all_available_materials()
{
  return 
  {
    "very smooth porcelain",
    "smooth porcelain",
    "porcelain",
    "ceramics",
    "rough ceramics",
    "glass",
    "imperfect glass",
    "frosted glass",
    "lambert"
  };
}

std::string MitsubaInterface::get_default_material()
{
  return "ceramics";
}

CamInfo MitsubaInterface::get_camera_from_scene_params(const std::vector<float> &scene_params)
{
  float fov_rad = scene_params.back();

  CamInfo camera;
  float h1 = 1.5;
  camera.fov_rad = fov_rad;
  float h2 = h1 * tan((3.14159265f / 3) / 2) / tan(camera.fov_rad / 2);
  camera.origin = float3(0, 0.5, h2);
  camera.target = float3(0, 0.5, 0);
  camera.up = float3(0, 1, 0);
  return camera;
}

std::vector<float> MitsubaInterface::get_default_scene_parameters()
{
  return {0,0,0,0,0,0, 100, 1000, 100, 100, 100, 0.01};
}