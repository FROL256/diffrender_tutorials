#pragma once
#include "virtual_drender.h"

struct DiffRenderMitsuba : public IDiffRender
{
  DiffRenderMitsuba() {};
  ~DiffRenderMitsuba() {};
  virtual void init(const DiffRenderSettings &settings) override
  {
    logerr("Mitsuba differentiable renderer not implemented. Use cmake option -DUSE_MITSUBA=ON");
  }; 
  virtual void commit(const Scene &scene) override
  {
    logerr("Mitsuba differentiable renderer not implemented. Use cmake option -DUSE_MITSUBA=ON");
  };
  virtual void render(const Scene &scene, const CamInfo* cams, Img *imgames, int a_viewsNum) override
  {
    logerr("Mitsuba differentiable renderer not implemented. Use cmake option -DUSE_MITSUBA=ON");
  };
  virtual void d_render(const Scene &scene, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                        DTriangleMesh &d_mesh,
                        Img* debugImages = nullptr, int debugImageNum = 0) override
  {
    logerr("Mitsuba differentiable renderer not implemented. Use cmake option -DUSE_MITSUBA=ON");
  };

  SHADING_MODEL mode;
private:
};