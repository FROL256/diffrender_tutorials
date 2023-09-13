#pragma once
#include "dmodels.h"
//#include "LiteMath.h"

namespace diff_render
{
  float3 sample_bilinear_clamp_3f(float2 tc, int w, int h, const float* data);
  void   sample_bilinear_clamp_3f_grad(float3 val, float2 tc, int w, int h, float* data);
}