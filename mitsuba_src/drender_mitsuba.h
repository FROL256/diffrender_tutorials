#pragma once
#include "virtual_drender.h"
#include "mitsuba_python_interaction.h"

namespace diff_render
{
struct DiffRenderMitsuba : public IDiffRender
{
  DiffRenderMitsuba();
  virtual ~DiffRenderMitsuba() override;
  virtual void init(const DiffRenderSettings &settings) override;
  virtual void commit(const Scene &scene) override;
  virtual void render(const Scene &scene, const CamInfo* cams, Img *imgames, int a_viewsNum) override;
  virtual void d_render(const Scene &scene, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                        DScene &d_mesh,
                        Img* debugImages = nullptr, int debugImageNum = 0) override;

  virtual float d_render_and_compare(const Scene &scene, const CamInfo* cams, const Img *target_images, int a_viewsNum, 
                                     const int edge_samples_in_total, DScene &d_mesh, Img* outImages = nullptr) override;
  SHADING_MODEL mode;
private:
  void init();
  bool optimization_inited = false;
  DFModel scene_to_dfmodel(const Scene &scene);
  MitsubaInterface mi;
  DiffRenderSettings settings;
};
}