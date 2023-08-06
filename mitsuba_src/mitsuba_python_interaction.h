#pragma once
#include "Python.h"
#include "virtual_drender.h"

typedef std::vector<std::pair<std::string, int>> PartOffsets;
typedef std::pair<std::vector<float>, PartOffsets> DFModel;

class MitsubaInterface
{
public:
  #define FLOAT_PER_VERTEX (3+3+2) //vec3 pos, vec3 norm, vec2 tc
  struct ModelLayout
  {
    //default layout is (pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, tc.x, tc.y)
    //if some value is -1, it means that model does not have such component in vertex
    ModelLayout(): ModelLayout(0, 3, 6, 8, 8) {}
    ModelLayout(int _p, int _n, int _tc, int _end, int _offset)
    {
      pos = _p;
      norm = _n;
      tc = _tc;
      end = _end;
      f_per_vert = _offset;
    }
    union
    {
      std::array<int, 4> offsets;
      struct
      {
        int pos;
        int norm;
        int tc;
        int end;
      };
    };
    int f_per_vert = 8;
  };
  enum MitsubaVariant
  {
    CUDA,
    LLVM
  };
  enum RenderStyle
  {
    SILHOUETTE,
    MONOCHROME,
    TEXTURED_CONST,
    MONOCHROME_DEMO,
    TEXTURED_DEMO
  };
  struct RenderSettings
  {
    RenderSettings() = default;
    RenderSettings(int iw, int ih, int spp, MitsubaVariant mv, RenderStyle rs, int _cameras_count = 1) : 
                   image_w(iw), image_h(ih), samples_per_pixel(spp), mitsubaVar(mv), renderStyle(rs) {};
    int image_w = 128;
    int image_h = 128;
    int samples_per_pixel = 16;
    MitsubaVariant mitsubaVar = MitsubaVariant::CUDA;
    RenderStyle renderStyle = RenderStyle::SILHOUETTE;
  };
  struct ModelInfo
  {
    //composite model is made from several parts
    //all parts have the same layout, but different parts
    //have different materials and textures
    struct PartInfo
    {
      std::string name = "main_part";
      std::string texture_name = "white.png";
      std::string material_name = "ceramics";
    };

    ModelLayout layout;
    std::vector<PartInfo> parts;

    ModelInfo() = default;

    PartInfo *get_part(const std::string &name)
    {
      for (auto &p : parts)
      {
        if (p.name == name)
          return &p;
      }
      return nullptr;
    }

    static ModelInfo simple_mesh(std::string texture_name, std::string material_name)
    {
      ModelInfo mi;
      mi.layout = ModelLayout();
      mi.parts.push_back(PartInfo{"main_part", texture_name, material_name});
      return mi;
    }
  };

  enum LossFunction
  {
    LOSS_MSE,
    LOSS_MSE_SQRT,
    LOSS_MIXED
  };

  MitsubaInterface(const std::string &scripts_dir, const std::string &file_name);
  ~MitsubaInterface();

  //basic mitsuba initialization, call before any other functions
  void init_scene_and_settings(RenderSettings render_settings, ModelInfo model_info);

  //initialize optimization cycle, set model to compare with. Set loss function and render settings for optimization cycle, includes init_scene_and_settings
  void init_optimization(const std::vector<std::string> &reference_image_dir, LossFunction loss_function,
                         RenderSettings render_settings, ModelInfo model_info,
                         bool save_intermediate_images = false);

  //WIP. initialize optimization of texture. includes init_scene_and_settings in it
  void init_optimization_with_tex(const std::vector<std::string> &reference_image_dir, LossFunction loss_function, 
                                  RenderSettings render_settings, ModelInfo model_info, 
                                  float texture_rec_learing_rate = 0.25,
                                  bool save_intermediate_images = false);
  //render model and save image to file, for debug purposes
  void render_model_to_file(const DFModel &model, const std::string &image_dir,
                            const CamInfo &camera, const std::vector<float> &scene_params);
  
  //renders model amd compare it with reference set by init_optimization function. Returns loss function value. Saves gradients
  //that are used by compute_final_grad
  float render_and_compare(const DFModel &model, const std::vector<CamInfo> &cameras, const std::vector<float> &scene_params,
                           double *timers = nullptr);

  //generator_jak size is [FLOAT_PER_VERTEX*params_count*vertex_count], final_grad size is [params_count]
  void compute_final_grad(const std::vector<float> &generator_jac, int params_count, int vertex_count, std::vector<float> &final_grad);
  void get_pos_derivatives(float *out_grad, int vertex_count);

  void finish();

  static std::vector<std::string> get_all_available_materials();
  static std::string get_default_material();
  static CamInfo get_camera_from_scene_params(const std::vector<float> &scene_params);
  static std::vector<float> get_default_scene_parameters();
//private:
  void show_errors();
  void set_model_max_size(int model_max_size);
  void init_optimization_internal(const std::string &function_name, const std::vector<std::string> &reference_image_dir,
                                  LossFunction loss_function, RenderSettings render_settings, ModelInfo model_info,
                                  float texture_rec_learing_rate, bool save_intermediate_images);
  int get_array_from_ctx_internal(const std::string &name, int buffer_id);//returns loaded array size (in floats)
  void set_array_to_ctx_internal(const std::string &name, int buffer_id, int size);//sends size float from buffer to mitsuba context 
  float render_and_compare_internal();//returns loss function value
  void model_to_ctx(const DFModel &model);
  void camera_to_ctx(const CamInfo &camera, std::string camera_name);
  void clear_buffer(int buffer_id, float val = 0);
  int get_camera_buffer_id()
  {
    return buffers.size() - 1;
  }
  int model_max_size = 0;
  int iteration = 0;
  std::vector<float *> buffers;
  std::vector<std::string> buffer_names;
  std::vector<int> active_parts;
  PyObject *pModule = nullptr, *mitsubaContext = nullptr;
  RenderSettings render_settings;
  ModelInfo model_info;
};