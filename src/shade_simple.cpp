
#include "dmodels.h"

inline std::vector<float> sample_bilinear_clamp(float2 tc, const CPUTexture &tex)
{
  tc *= float2(tex.w, tex.h);
  int2 tc0 = clamp(int2(tc), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  int2 tc1 = clamp(int2(tc) + int2(1, 1), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  float2 dtc = tc - float2(tc0);
  const float *p00 = tex.get(tc0.x, tc0.y);
  const float *p01 = tex.get(tc0.x, tc1.y);
  const float *p10 = tex.get(tc1.x, tc0.y);
  const float *p11 = tex.get(tc1.x, tc1.y);

  std::vector<float> res(tex.channels, 0);
  for (int i = 0; i < tex.channels; i++)
  {
    res[i] = (1 - dtc.x) * (1 - dtc.y) * p00[i] + (1 - dtc.x) * dtc.y * p01[i] + dtc.x * (1 - dtc.y) * p10[i] + dtc.x * dtc.y * p11[i];
  }

  return res;
}

template <>
float3 shade<MATERIAL::SILHOUETTE>(const TriangleMesh &mesh, const SurfaceInfo &surfInfo, const float3 ray_pos, const float3 ray_dir)
{
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0);
  else
    return float3(1, 1, 1);
}

template <>
void shade_grad<MATERIAL::SILHOUETTE>(const TriangleMesh &mesh, const SurfaceInfo &surfElem, const float3 ray_pos, const float3 ray_dir,
                                        const float3 val, const AuxData aux, DTriangleMesh &grad)
{

}

template <>
float3 shade<MATERIAL::VERTEX_COLOR>(const TriangleMesh &mesh, const SurfaceInfo &surfInfo, const float3 ray_pos, const float3 ray_dir)
{
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = mesh.indices[surfInfo.faceId * 3 + 0];
  const auto B = mesh.indices[surfInfo.faceId * 3 + 1];
  const auto C = mesh.indices[surfInfo.faceId * 3 + 2];
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  return mesh.colors[A] * (1.0f - u - v) + mesh.colors[B] * v + u * mesh.colors[C];
}

template <>
void shade_grad<MATERIAL::VERTEX_COLOR>(const TriangleMesh &mesh, const SurfaceInfo &surfElem, const float3 ray_pos, const float3 ray_dir,
                                          const float3 val, const AuxData aux, DTriangleMesh &grad)
{
  auto A = mesh.indices[surfElem.faceId * 3 + 0];
  auto B = mesh.indices[surfElem.faceId * 3 + 1];
  auto C = mesh.indices[surfElem.faceId * 3 + 2];

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

  const float3 c0 = mesh.colors[A];
  const float3 c1 = mesh.colors[B];
  const float3 c2 = mesh.colors[C];
  const float dF_dU = dot((c2 - c0), val);
  const float dF_dV = dot((c1 - c0), val);

  if (dF_dU * dF_dU > 0.0f || dF_dV * dF_dV > 0.0f)
  {
    const float3 v0 = mesh.vertices[A];
    const float3 v1 = mesh.vertices[B];
    const float3 v2 = mesh.vertices[C];
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
float3 shade<MATERIAL::DIFFUSE>(const TriangleMesh &mesh, const SurfaceInfo &surfInfo, const float3 ray_pos, const float3 ray_dir)
{
  if (surfInfo.faceId == unsigned(-1))
    return float3(0, 0, 0); // BGCOLOR

  const auto A = mesh.indices[surfInfo.faceId * 3 + 0];
  const auto B = mesh.indices[surfInfo.faceId * 3 + 1];
  const auto C = mesh.indices[surfInfo.faceId * 3 + 2];
  const float u = surfInfo.u;
  const float v = surfInfo.v;

  float2 tc = mesh.tc[A] * (1.0f - u - v) + mesh.tc[B] * v + u * mesh.tc[C];
  auto res = sample_bilinear_clamp(tc, mesh.textures[0]);
  return float3(res[0], res[1], res[2]);
}

template <>
void shade_grad<MATERIAL::DIFFUSE>(const TriangleMesh &mesh, const SurfaceInfo &surfElem, const float3 ray_pos, const float3 ray_dir,
                                     const float3 val, const AuxData aux, DTriangleMesh &grad)
{
  auto A = mesh.indices[surfElem.faceId * 3 + 0];
  auto B = mesh.indices[surfElem.faceId * 3 + 1];
  auto C = mesh.indices[surfElem.faceId * 3 + 2];

  const float u = surfElem.u;
  const float v = surfElem.v;

  auto &tex = mesh.textures[0];

  float2 tc = mesh.tc[A] * (1.0f - u - v) + mesh.tc[B] * v + u * mesh.tc[C];
  tc *= float2(tex.w, tex.h);
  int2 tc0 = clamp(int2(tc), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  int2 tc1 = clamp(int2(tc) + int2(1, 1), int2(0, 0), int2(tex.w - 1, tex.h - 1));
  float2 dtc = tc - float2(tc0);
  int off = grad.tex_offset(0);

  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y)] += (1 - dtc.x) * (1 - dtc.y) * val.x;
  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y) + 1] += (1 - dtc.x) * (1 - dtc.y) * val.y;
  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y) + 2] += (1 - dtc.x) * (1 - dtc.y) * val.z;

  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y)] += (1 - dtc.x) * dtc.y * val.x;
  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y) + 1] += (1 - dtc.x) * dtc.y * val.y;
  grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y) + 2] += (1 - dtc.x) * dtc.y * val.z;

  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y)] += dtc.x * (1 - dtc.y) * val.x;
  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y) + 1] += dtc.x * (1 - dtc.y) * val.y;
  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y) + 2] += dtc.x * (1 - dtc.y) * val.z;

  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y)] += dtc.x * dtc.y * val.x;
  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y) + 1] += dtc.x * dtc.y * val.y;
  grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y) + 2] += dtc.x * dtc.y * val.z;
}