#pragma once
#include "dmesh.h"

struct IOptimizer
{
  virtual void         Init(const TriangleMesh& a_mesh, const Img& a_image) = 0;
  virtual TriangleMesh Run (size_t a_numIters = 100) = 0;
};
