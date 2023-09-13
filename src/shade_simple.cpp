
#include "dmodels.h"
#include "shade_common.h"
namespace diff_render
{
template <>
float3 shade<SHADING_MODEL::SILHOUETTE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.primId == unsigned(-1))
    return float3(0, 0, 0);
  else
    return float3(1, 1, 1);
}

template <>
void shade_grad<SHADING_MODEL::SILHOUETTE>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                           const float3 val, const AuxData aux, DScene &grad)
{

}

template <>
float3 shade<SHADING_MODEL::VERTEX_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.primId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 0);
  const auto B = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 1);
  const auto C = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 2);
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  return scene.get_color(A) * (1.0f - u - v) + scene.get_color(B) * v + u * scene.get_color(C);
}

template <>
void shade_grad<SHADING_MODEL::VERTEX_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                             const float3 val, const AuxData aux, DScene &grad)
{
  float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y, &ray_pos, &ray_dir);
  if (surfInfo.primId == unsigned(-1))
    return;

  const auto A = scene.get_vertex_n(surfInfo.geomId, surfInfo.primId * 3 + 0);
  const auto B = scene.get_vertex_n(surfInfo.geomId, surfInfo.primId * 3 + 1);
  const auto C = scene.get_vertex_n(surfInfo.geomId, surfInfo.primId * 3 + 2);

  const float u = surfInfo.u;
  const float v = surfInfo.v;

  auto contribA = (1.0f - u - v) * val;
  auto contribB = v * val;
  auto contribC = u * val;

  auto *dm_ptr = grad.get_dmesh(surfInfo.geomId);
  if (dm_ptr)
  {
    auto &dm = *dm_ptr;
    dm.color(A)[0] += GradReal(contribA.x);
    dm.color(A)[1] += GradReal(contribA.y);
    dm.color(A)[2] += GradReal(contribA.z);

    dm.color(B)[0] += GradReal(contribB.x);
    dm.color(B)[1] += GradReal(contribB.y);
    dm.color(B)[2] += GradReal(contribB.z);

    dm.color(C)[0] += GradReal(contribC.x);
    dm.color(C)[1] += GradReal(contribC.y);
    dm.color(C)[2] += GradReal(contribC.z);

  /*
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

      dm.pos(A)[0] += GradReal(contribVA.x);
      dm.pos(A)[1] += GradReal(contribVA.y);
      dm.pos(A)[2] += GradReal(contribVA.z);

      dm.pos(B)[0] += GradReal(contribVB.x);
      dm.pos(B)[1] += GradReal(contribVB.y);
      dm.pos(B)[2] += GradReal(contribVB.z);

      dm.pos(C)[0] += GradReal(contribVC.x);
      dm.pos(C)[1] += GradReal(contribVC.y);
      dm.pos(C)[2] += GradReal(contribVC.z);
    }
  */
  }
}

template <>
float3 shade<SHADING_MODEL::TEXTURE_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.primId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  //logerr("%d %d %d",surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 0);
  const auto A = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 0);
  const auto B = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 1);
  const auto C = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 2);
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  float2 tc = scene.get_tc(A) * (1.0f - u - v) + scene.get_tc(B) * v + u * scene.get_tc(C);
  const auto& tex = scene.get_tex(surfInfo.geomId, 0);
  return sample_bilinear_clamp_3f(tc, tex.w, tex.h, tex.data());
}

template <>
void shade_grad<SHADING_MODEL::TEXTURE_COLOR>(const Scene &scene, IRayTracer *m_pTracer, const float2 screen_pos,
                                              const float3 val, const AuxData aux, DScene &grad)
{
  SurfaceInfo surfInfo = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y);
  if (surfInfo.primId == unsigned(-1))
    return;
  
  auto *dm_ptr = grad.get_dmesh(surfInfo.geomId);
  if (dm_ptr)
  {
    auto &dm = *dm_ptr;
    GradReal*  texdata = dm.tex_by_id(0);
    const int3 texinfo = dm.tex_info_by_id(0);

    const auto A = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 0);
    const auto B = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 1);
    const auto C = scene.get_index(surfInfo.geomId, surfInfo.instId, surfInfo.primId * 3 + 2);
  
    const float u = surfInfo.u;
    const float v = surfInfo.v;
                                          // we actually need 'tex' only to get width and height for our texture
    const int    w  = texinfo.x;
    const int    h  = texinfo.y;
    const int  chan = texinfo.z;
    assert(chan == 3);

    const float2 tc = scene.get_tc(A) * (1.0f - u - v) + scene.get_tc(B) * v + u * scene.get_tc(C);
    diff_render::sample_bilinear_clamp_3f_grad(val, tc, w, h, texdata);
  }
}



}