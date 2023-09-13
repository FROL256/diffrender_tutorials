#include "shade_common.h"

LiteMath::float3 diff_render::sample_bilinear_clamp_3f(float2 tc, int w, int h, const float* data)
{
  tc = clamp(tc, float2(0,0), float2(1,1));
  tc *= float2(w, h);
  
  const int2   tc0 = clamp(int2(tc), int2(0, 0), int2(w - 1, h - 1));
  const int2   tc1 = clamp(int2(tc) + int2(1, 1), int2(0, 0), int2(w - 1, h - 1));
  const float2 dtc = tc - float2(tc0);
  
  const float *p00 = data + 3*(tc0.y*w + tc0.x); // tex.get(tc0.x, tc0.y);
  const float *p01 = data + 3*(tc1.y*w + tc0.x); // tex.get(tc0.x, tc1.y);
  const float *p10 = data + 3*(tc0.y*w + tc1.x); // tex.get(tc1.x, tc0.y);
  const float *p11 = data + 3*(tc1.y*w + tc1.x); // tex.get(tc1.x, tc1.y);

  float3 res;
  for (int i = 0; i < 3; i++)
    res[i] = (1 - dtc.x) * (1 - dtc.y) * p00[i] + (1 - dtc.x) * dtc.y * p01[i] + dtc.x * (1 - dtc.y) * p10[i] + dtc.x * dtc.y * p11[i];
  return res;
}

void diff_render::sample_bilinear_clamp_3f_grad(LiteMath::float3 val, LiteMath::float2 tc, int w, int h, float* data)
{
  const float2 px  = tc*float2(w, h);
  const int2 tc0   = clamp(int2(px), int2(0, 0), int2(w - 1, h - 1));
  const int2 tc1   = clamp(int2(px) + int2(1, 1), int2(0, 0), int2(w - 1, h - 1));
  const float2 dtc = px - float2(tc0);

  constexpr int chan = 3; // optimized loop for chan==3
  data[chan*(tc0.y*w + tc0.x) + 0] += (1 - dtc.x) * (1 - dtc.y) * val.x;
  data[chan*(tc0.y*w + tc0.x) + 1] += (1 - dtc.x) * (1 - dtc.y) * val.y;
  data[chan*(tc0.y*w + tc0.x) + 2] += (1 - dtc.x) * (1 - dtc.y) * val.z; 

  data[chan*(tc1.y*w + tc0.x) + 0] += (1 - dtc.x) * dtc.y * val.x;
  data[chan*(tc1.y*w + tc0.x) + 1] += (1 - dtc.x) * dtc.y * val.y;
  data[chan*(tc1.y*w + tc0.x) + 2] += (1 - dtc.x) * dtc.y * val.z;

  data[chan*(tc0.y*w + tc1.x) + 0] += dtc.x * (1 - dtc.y) * val.x;
  data[chan*(tc0.y*w + tc1.x) + 1] += dtc.x * (1 - dtc.y) * val.y;
  data[chan*(tc0.y*w + tc1.x) + 2] += dtc.x * (1 - dtc.y) * val.z;

  data[chan*(tc1.y*w + tc1.x) + 0] += dtc.x * dtc.y * val.x;
  data[chan*(tc1.y*w + tc1.x) + 1] += dtc.x * dtc.y * val.y;
  data[chan*(tc1.y*w + tc1.x) + 2] += dtc.x * dtc.y * val.z;
}