#pragma once
#include "dmesh.h"

enum OPT_ALGORITHM{GD_Naive=1, GD_AdaGrad=2, GD_RMSProp=3, GD_Adam=4};

struct OptimizerParameters
{
  int decayPeriod   = 30;
  OPT_ALGORITHM alg = GD_Naive;
};

struct IOptimizer
{
  virtual void         Init(const TriangleMesh& a_mesh, const Img& a_image, OptimizerParameters a_params = OptimizerParameters()) = 0;
  virtual TriangleMesh Run (size_t a_numIters = 100) = 0;
};
