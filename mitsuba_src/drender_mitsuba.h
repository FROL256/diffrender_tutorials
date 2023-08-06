#pragma once
#include "virtual_drender.h"
#include "mitsuba_python_interaction.h"

struct DiffRenderMitsuba : public IDiffRender
{
  DiffRenderMitsuba();
  ~DiffRenderMitsuba();
  virtual void init(const DiffRenderSettings &settings) override;
  virtual void commit(const Scene &scene) override;
  virtual void render(const Scene &scene, const CamInfo* cams, Img *imgames, int a_viewsNum) override;
  virtual void d_render(const Scene &scene, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                        DTriangleMesh &d_mesh,
                        Img* debugImages = nullptr, int debugImageNum = 0) override;

  SHADING_MODEL mode;
private:
  void init();
  DFModel scene_to_dfmodel(const Scene &scene);
  MitsubaInterface mi;
  DiffRenderSettings settings;
};