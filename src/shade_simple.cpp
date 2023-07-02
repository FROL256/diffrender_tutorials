
#include "dmodels.h"
#include "shade_common.h"

template <>
float3 shade<SHADING_MODEL::SILHOUETTE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0);
  else
    return float3(1, 1, 1);
}

template <>
void shade_grad<SHADING_MODEL::SILHOUETTE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                           const float3 val, const AuxData aux, DTriangleMesh &grad)
{

}

template <>
float3 shade<SHADING_MODEL::VERTEX_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = scene.get_index(surfInfo.faceId * 3 + 0);
  const auto B = scene.get_index(surfInfo.faceId * 3 + 1);
  const auto C = scene.get_index(surfInfo.faceId * 3 + 2);
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  return scene.get_color(A) * (1.0f - u - v) + scene.get_color(B) * v + u * scene.get_color(C);
}

template <>
void shade_grad<SHADING_MODEL::VERTEX_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                             const float3 val, const AuxData aux, DTriangleMesh &grad)
{
  float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
  SurfaceInfo surfElem = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y, &ray_pos, &ray_dir);
  if (surfElem.faceId == unsigned(-1))
    return;

  const auto A = scene.get_index(surfElem.faceId * 3 + 0);
  const auto B = scene.get_index(surfElem.faceId * 3 + 1);
  const auto C = scene.get_index(surfElem.faceId * 3 + 2);

  const float u = surfElem.u;
  const float v = surfElem.v;

  GradReal *d_colors = grad.colors_s();
  GradReal *d_pos = grad.vertices_s();

  auto contribA = (1.0f - u - v) * val;
  auto contribB = v * val;
  auto contribC = u * val;

  d_colors[A * 3 + 0] += GradReal(contribA.x);
  d_colors[A * 3 + 1] += GradReal(contribA.y);
  d_colors[A * 3 + 2] += GradReal(contribA.z);

  d_colors[B * 3 + 0] += GradReal(contribB.x);
  d_colors[B * 3 + 1] += GradReal(contribB.y);
  d_colors[B * 3 + 2] += GradReal(contribB.z);

  d_colors[C * 3 + 0] += GradReal(contribC.x);
  d_colors[C * 3 + 1] += GradReal(contribC.y);
  d_colors[C * 3 + 2] += GradReal(contribC.z);

  const float3 c0 = scene.get_color(A);
  const float3 c1 = scene.get_color(B);
  const float3 c2 = scene.get_color(C);
  const float dF_dU = dot((c2 - c0), val);
  const float dF_dV = dot((c1 - c0), val);

  if (dF_dU * dF_dU > 0.0f || dF_dV * dF_dV > 0.0f)
  {
    const float3 v0 = scene.get_pos(A);
    const float3 v1 = scene.get_pos(B);
    const float3 v2 = scene.get_pos(C);
    float3 dU_dvert[3] = {};
    float3 dV_dvert[3] = {};

    BarU_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, dU_dvert[0].M, dU_dvert[1].M, dU_dvert[2].M);
    BarV_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, dV_dvert[0].M, dV_dvert[1].M, dV_dvert[2].M);

    auto contribVA = (dF_dU * dU_dvert[0] + dF_dV * dV_dvert[0]);
    auto contribVB = (dF_dU * dU_dvert[1] + dF_dV * dV_dvert[1]);
    auto contribVC = (dF_dU * dU_dvert[2] + dF_dV * dV_dvert[2]);

    d_pos[A * 3 + 0] += GradReal(contribVA.x);
    d_pos[A * 3 + 1] += GradReal(contribVA.y);
    d_pos[A * 3 + 2] += GradReal(contribVA.z);

    d_pos[B * 3 + 0] += GradReal(contribVB.x);
    d_pos[B * 3 + 1] += GradReal(contribVB.y);
    d_pos[B * 3 + 2] += GradReal(contribVB.z);

    d_pos[C * 3 + 0] += GradReal(contribVC.x);
    d_pos[C * 3 + 1] += GradReal(contribVC.y);
    d_pos[C * 3 + 2] += GradReal(contribVC.z);
  }
}

template <>
float3 shade<SHADING_MODEL::DIFFUSE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = scene.get_index(surfInfo.faceId * 3 + 0);
  const auto B = scene.get_index(surfInfo.faceId * 3 + 1);
  const auto C = scene.get_index(surfInfo.faceId * 3 + 2);
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  float2 tc = scene.get_tc(A) * (1.0f - u - v) + scene.get_tc(B) * v + u * scene.get_tc(C);
  auto res = sample_bilinear_clamp(tc, scene.get_tex(0));
  return float3(res[0], res[1], res[2]);
}

template <>
void shade_grad<SHADING_MODEL::DIFFUSE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                        const float3 val, const AuxData aux, DTriangleMesh &grad)
{
  SurfaceInfo surfElem = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfElem.faceId == unsigned(-1))
    return;
  
  const auto A = scene.get_index(surfElem.faceId * 3 + 0);
  const auto B = scene.get_index(surfElem.faceId * 3 + 1);
  const auto C = scene.get_index(surfElem.faceId * 3 + 2);

  const float u = surfElem.u;
  const float v = surfElem.v;

  auto &tex = scene.get_tex(0);

  float2 tc = scene.get_tc(A) * (1.0f - u - v) + scene.get_tc(B) * v + u * scene.get_tc(C);
  tc *= float2(tex.w, tex.h);
  int2 tc0 = clamp(int2(tc), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  int2 tc1 = clamp(int2(tc) + int2(1, 1), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  float2 dtc = tc - float2(tc0);
  int off = grad.tex_offset(0);

  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc0.y)] += (1 - dtc.x) * (1 - dtc.y) * val.x;
  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc0.y) + 1] += (1 - dtc.x) * (1 - dtc.y) * val.y;
  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc0.y) + 2] += (1 - dtc.x) * (1 - dtc.y) * val.z;

  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc1.y)] += (1 - dtc.x) * dtc.y * val.x;
  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc1.y) + 1] += (1 - dtc.x) * dtc.y * val.y;
  grad[off + scene.get_tex(0).pixel_to_offset(tc0.x, tc1.y) + 2] += (1 - dtc.x) * dtc.y * val.z;

  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc0.y)] += dtc.x * (1 - dtc.y) * val.x;
  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc0.y) + 1] += dtc.x * (1 - dtc.y) * val.y;
  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc0.y) + 2] += dtc.x * (1 - dtc.y) * val.z;

  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc1.y)] += dtc.x * dtc.y * val.x;
  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc1.y) + 1] += dtc.x * dtc.y * val.y;
  grad[off + scene.get_tex(0).pixel_to_offset(tc1.x, tc1.y) + 2] += dtc.x * dtc.y * val.z;
}