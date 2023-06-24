#include "drender.h"

// build a discrete CDF using edge length
Sampler build_edge_sampler(const TriangleMesh &mesh, const std::vector<Edge> &edges) 
{
  std::vector<float> pmf;
  std::vector<float> cdf;
  pmf.reserve(edges.size());
  cdf.reserve(edges.size() + 1);
  cdf.push_back(0);
  for (auto edge : edges) {
      auto v0 = mesh.vertices[edge.v0];
      auto v1 = mesh.vertices[edge.v1];
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
std::vector<Edge> collect_edges(const TriangleMesh &mesh) 
{
  std::set<Edge> edges;
  for (size_t i=0; i<mesh.indices.size();i+=3) 
  {
    auto A = mesh.indices[i+0];
    auto B = mesh.indices[i+1];
    auto C = mesh.indices[i+2];  
    edges.insert(Edge(A, B));
    edges.insert(Edge(B, C));
    edges.insert(Edge(C, A));
  }
  return std::vector<Edge>(edges.begin(), edges.end());
}

inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux,
                      std::vector<GradReal> &d_pos)
{
  float3 v0_d[2] = {{0, 0, 0}, {0, 0, 0}};
  float3 v1_d[2] = {{0, 0, 0}, {0, 0, 0}};

  float3 v0_3d = mesh.vertices[v0];
  float3 v1_3d = mesh.vertices[v1];

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

std::shared_ptr<IDiffRender> MakeDifferentialRenderer(const TriangleMesh &a_mesh, int a_samplesPerPixel)
{
  switch (a_mesh.material)
  {
  case MATERIAL::SILHOUETTE:
    {
    auto impl = std::make_shared<DiffRender<MATERIAL::SILHOUETTE>>();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
    }
    break;    
  case MATERIAL::VERTEX_COLOR:
    {
    auto impl = std::make_shared<DiffRender<MATERIAL::VERTEX_COLOR>>();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
    }
    break;
  case MATERIAL::DIFFUSE:
    {
    auto impl = std::make_shared<DiffRender<MATERIAL::DIFFUSE>>();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
    }
    break;
  case MATERIAL::LAMBERT:
    {
    auto impl = std::make_shared<DiffRender<MATERIAL::LAMBERT>>();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
    }
    break;
  default:
    assert(false);
    break;
  }
  return nullptr;
}
