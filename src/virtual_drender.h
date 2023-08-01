#pragma once
#include "scene.h"
#include "dmesh.h"
#include "camera.h"

struct IDiffRender
{
  virtual void commit(const Scene &scene) = 0;
  virtual void render(const Scene &scene, const CamInfo* cams, Img *imgames, int a_viewsNum) = 0;
  virtual void d_render(const Scene &scene, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                        DTriangleMesh &d_mesh,
                        Img* debugImages = nullptr, int debugImageNum = 0) = 0;

  SHADING_MODEL mode;
};