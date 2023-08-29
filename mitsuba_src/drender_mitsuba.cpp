#include "drender_mitsuba.h"
#include "mitsuba_python_interaction.h"

namespace diff_render
{
void DiffRenderMitsuba::init(const DiffRenderSettings &_settings)
{
  settings = _settings;
  mode = settings.mode;
}

DiffRenderMitsuba::DiffRenderMitsuba():
mi("scripts", "mitsuba_optimization_embedded")
{

}

DiffRenderMitsuba::~DiffRenderMitsuba()
{

}

void DiffRenderMitsuba::commit(const Scene &scene)
{
  scene.prepare_for_render();
}

DFModel DiffRenderMitsuba::scene_to_dfmodel(const Scene &scene)
{
  PartOffsets po = {{"main_part",0}};
  ::std::vector<float> m;
  for (int i=0;i<scene.get_meshes().size();i++)
  {
    auto &mesh = scene.get_mesh(i);
    int off = m.size();
    int v_cnt = mesh.indices.size();
    m.resize(off + v_cnt*FLOAT_PER_VERTEX);
    for (auto j : mesh.indices)
    {
      m[off + 0] = mesh.vertices[j].x;
      m[off + 1] = mesh.vertices[j].y;
      m[off + 2] = mesh.vertices[j].z;

      if (mesh.normals.size() > j)
      {
        m[off + 3] = mesh.normals[j].x;
        m[off + 4] = mesh.normals[j].y;
        m[off + 5] = mesh.normals[j].z;
      }
      else
      {
        m[off + 3] = 1;
        m[off + 4] = 0;
        m[off + 5] = 0;
      }
      if (mesh.tc.size() > j)
      {
        m[off + 6] = mesh.tc[j].x;
        m[off + 7] = mesh.tc[j].y;
      }
      else
      {
        m[off + 6] = 0;
        m[off + 7] = 0;
      }
      off += FLOAT_PER_VERTEX;
    }
  }
  return {m, po};
}

void DiffRenderMitsuba::render(const Scene &scene, const CamInfo* cams, Img *imgames, int a_viewsNum)
{
  DFModel model = scene_to_dfmodel(scene);
  
  for (int i = 0; i < a_viewsNum; i++)
  {
    const CamInfo &camera = cams[i];
    Img &res_image = imgames[i];

    MitsubaInterface::RenderSettings rs;
    rs.image_w = camera.width;
    rs.image_h = camera.height;
    rs.renderStyle = settings.mode == SHADING_MODEL::SILHOUETTE ? MitsubaInterface::RenderStyle::SILHOUETTE : MitsubaInterface::RenderStyle::TEXTURED_DEMO;
    rs.mitsubaVar = MitsubaInterface::MitsubaVariant::CUDA;
    rs.samples_per_pixel = settings.spp;

    ::std::string filename = "output/mitsuba_images/tmp"+ ::std::to_string(i)+".png";
    mi.init_scene_and_settings(rs, MitsubaInterface::ModelInfo::simple_mesh("white.png", mi.get_default_material()));
    mi.render_model_to_file(model, filename, camera, mi.get_default_scene_parameters());
    res_image = LiteImage::LoadImage<float3>(filename.c_str());
  }
}

void DiffRenderMitsuba::d_render(const Scene &scene, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                                 DScene &d_mesh,
                                 Img* debugImages, int debugImageNum)
{
  
}
float DiffRenderMitsuba::d_render_and_compare(const Scene &scene, const CamInfo* cams, const Img *target_images, int a_viewsNum, 
                                              const int edge_samples_in_total, DScene &d_mesh, Img* outImages)
{
  if (!optimization_inited)
  {
    optimization_inited = true;

    //save target images
    ::std::vector<::std::string> target_image_dirs;
    for (int i = 0; i < a_viewsNum; i++)
    {
      ::std::string dir = "output/mitsuba_images/target"+ ::std::to_string(i)+".png";
      target_image_dirs.push_back(dir);
      LiteImage::SaveImage(dir.c_str(), target_images[i]);
    }

    //set render settings
    const CamInfo &camera = cams[0];
    MitsubaInterface::RenderSettings rs;
    rs.image_w = camera.width;
    rs.image_h = camera.height;
    rs.renderStyle = settings.mode == SHADING_MODEL::SILHOUETTE ? MitsubaInterface::RenderStyle::SILHOUETTE : MitsubaInterface::RenderStyle::TEXTURED_DEMO;
    rs.mitsubaVar = MitsubaInterface::MitsubaVariant::CUDA;
    rs.samples_per_pixel = settings.spp;

    mi.init_optimization(target_image_dirs, MitsubaInterface::LOSS_MSE, rs, 
                         MitsubaInterface::ModelInfo::simple_mesh("white.png", mi.get_default_material()));
  }
  DFModel model = scene_to_dfmodel(scene);
  ::std::vector<CamInfo> cameras(cams, cams + a_viewsNum);
  float mse = mi.render_and_compare(model, cameras, mi.get_default_scene_parameters());
  
  d_mesh.clear();
  mi.get_pos_derivatives(d_mesh.get_dmeshes()[0].pos(0), d_mesh.get_dmeshes()[0].vertex_count());
  return mse;
}                 
}