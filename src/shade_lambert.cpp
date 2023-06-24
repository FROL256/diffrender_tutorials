#include "dmodels.h"
#include "shade_common.h"

template <>
float3 shade<MATERIAL::LAMBERT>(const TriangleMesh &mesh, IRayTracer *m_pTracer, const float2 screen_pos)
{
  //TODO:move to scene
  const float3 light_dir = normalize(float3(-1,-1,0));
  const float3 light_color = float3(1,0.5,0.2);
  const float3 ambient_light_color = 0.1f*float3(1,1,1);
  const float BIAS = 1e-5;

  float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y, &ray_pos, &ray_dir);
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = mesh.indices[surfInfo.faceId * 3 + 0];
  const auto B = mesh.indices[surfInfo.faceId * 3 + 1];
  const auto C = mesh.indices[surfInfo.faceId * 3 + 2];
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  float2 tc = mesh.tc[A] * (1.0f - u - v) + mesh.tc[B] * v + u * mesh.tc[C];
  float3 n = mesh.normals[A] * (1.0f - u - v) + mesh.normals[B] * v + u * mesh.normals[C];
  auto diffuse_v = sample_bilinear_clamp(tc, mesh.textures[0]);
  float3 diffuse = float3(diffuse_v[0], diffuse_v[1], diffuse_v[2]);

  float3 surf_pos = ray_pos + (surfInfo.t-BIAS)*ray_dir;
  float shade = m_pTracer->GetNearestHit(surf_pos, -1.0f*light_dir).faceId == unsigned(-1) ? 1 : 0;
  return (shade*light_color*std::max(0.0f,dot(n,-1.0f*light_dir)) + ambient_light_color)*diffuse;
}

template <>
void shade_grad<MATERIAL::LAMBERT>(const TriangleMesh &mesh, IRayTracer *m_pTracer, const float2 screen_pos,
                                     const float3 val, const AuxData aux, DTriangleMesh &grad)
{
  shade_grad<MATERIAL::DIFFUSE>(mesh, m_pTracer, screen_pos, val, aux, grad);
}