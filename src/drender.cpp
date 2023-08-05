#include "drender.h"

// build a discrete CDF using edge length
Sampler build_edge_sampler(const Scene &scene, const std::vector<Edge> &edges) 
{
  std::vector<float> pmf;
  std::vector<float> cdf;
  pmf.reserve(edges.size());
  cdf.reserve(edges.size() + 1);
  cdf.push_back(0);
  for (auto edge : edges) {
      auto v0 = scene.get_pos(edge.v0);
      auto v1 = scene.get_pos(edge.v1);
      pmf.push_back(length(v1 - v0));
      cdf.push_back(pmf.back() + cdf.back());
  }
  auto length_sum = cdf.back();
  std::for_each(pmf.begin(), pmf.end(), [&](float &p) {p /= length_sum;});
  std::for_each(cdf.begin(), cdf.end(), [&](float &p) {p /= length_sum;});
  return Sampler{pmf, cdf};
}

// binary search for inverting the CDF in the sampler
int sample(const Sampler &sampler, const float u) 
{
  auto cdf = sampler.cdf;
  return clamp(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin() - 1, 0, cdf.size() - 2);
}

// given a triangle mesh, collect all edges.
std::vector<Edge> collect_edges(const Scene &scene) 
{
  std::set<Edge> edges;
  for (size_t i=0; i<scene.indices_size();i+=3) 
  {
    auto A = scene.get_index(i);
    auto B = scene.get_index(i+1);
    auto C = scene.get_index(i+2); 
    edges.insert(Edge(A, B));
    edges.insert(Edge(B, C));
    edges.insert(Edge(C, A));
  }
  return std::vector<Edge>(edges.begin(), edges.end());
}

inline void edge_grad(const Scene &scene, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux,
                      std::vector<GradReal> &d_pos)
{
  float3 v0_d[2] = {{0, 0, 0}, {0, 0, 0}};
  float3 v1_d[2] = {{0, 0, 0}, {0, 0, 0}};

  float3 v0_3d = scene.get_pos(v0);
  float3 v1_3d = scene.get_pos(v1);

  VS_X_grad(v0_3d.M, *(aux.pCamInfo), v0_d[0].M);
  VS_Y_grad(v0_3d.M, *(aux.pCamInfo), v0_d[1].M);
  VS_X_grad(v1_3d.M, *(aux.pCamInfo), v1_d[0].M);
  VS_Y_grad(v1_3d.M, *(aux.pCamInfo), v1_d[1].M);

  const float dv0_dx = v0_d[0].x * d_v0.x; // + v0_dx.y*d_v0.y; ==> 0
  const float dv0_dy = v0_d[1].y * d_v0.y; // + v0_dy.x*d_v0.x; ==> 0
  const float dv0_dz = (v0_d[0].z * d_v0.x + v0_d[1].z * d_v0.y);

  const float dv1_dx = v1_d[0].x * d_v1.x; // + v1_dx.y*d_v1.y; ==> 0
  const float dv1_dy = v1_d[1].y * d_v1.y; // + v1_dy.x*d_v1.x; ==> 0
  const float dv1_dz = (v1_d[0].z * d_v1.x + v1_d[1].z * d_v1.y);

  d_pos[v0 * 3 + 0] += GradReal(dv0_dx);
  d_pos[v0 * 3 + 1] += GradReal(dv0_dy);
  d_pos[v0 * 3 + 2] += GradReal(dv0_dz);

  d_pos[v1 * 3 + 0] += GradReal(dv1_dx);
  d_pos[v1 * 3 + 1] += GradReal(dv1_dy);
  d_pos[v1 * 3 + 2] += GradReal(dv1_dz);
}

std::shared_ptr<IDiffRender> MakeDifferentialRenderer(const Scene &scene, const DiffRenderSettings &settings)
{
  switch (settings.mode)
  {
  case SHADING_MODEL::SILHOUETTE:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::SILHOUETTE>>();
    impl->init(settings);
    return impl;
    }
    break;    
  case SHADING_MODEL::VERTEX_COLOR:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::VERTEX_COLOR>>();
    impl->init(settings);
    return impl;
    }
    break;
  case SHADING_MODEL::DIFFUSE:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::DIFFUSE>>();
    impl->init(settings);
    return impl;
    }
    break;
  case SHADING_MODEL::LAMBERT:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::LAMBERT>>();
    impl->init(settings);
    return impl;
    }
    break;
  case SHADING_MODEL::PHONG:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::PHONG>>();
    impl->init(settings);
    return impl;
    }
    break;
  case SHADING_MODEL::GGX:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::GGX>>();
    impl->init(settings);
    return impl;
    }
    break;
  case SHADING_MODEL::PATH_TEST:
    {
    auto impl = std::make_shared<DiffRender<SHADING_MODEL::PATH_TEST>>();
    impl->init(settings);
    return impl;
    }
    break;
  default:
    assert(false);
    break;
  }
  return nullptr;
}
