#pragma once

#include "LiteMath.h"
#include "dmesh.h"
#include "functions.h"
#include <cstdio>

#define DEBUG_RENDER 0
constexpr static int  G_DEBUG_VERT_ID = 0;

struct AuxData
{
  const CamInfo* pCamInfo = nullptr;
  Img* debugImages  = nullptr;
  int debugImageNum = 0;
};

template<MATERIAL material>
float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir);

template<MATERIAL material>
void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, 
                         const float3 val, const AuxData aux, DTriangleMesh& grad);
