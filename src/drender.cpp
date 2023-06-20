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

std::shared_ptr<IDiffRender> MakeDifferentialRenderer(const TriangleMesh &a_mesh, int a_samplesPerPixel)
{
  if(a_mesh.m_meshType == MODELS::TRIANGLE3D_VERT_COLOR::getMeshType())
  {
    auto impl = std::make_shared< DiffRender<MODELS::TRIANGLE3D_VERT_COLOR> >();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
  }

  if(a_mesh.m_meshType == MODELS::TRIANGLE3D_TEXTURED::getMeshType())
  {
    auto impl = std::make_shared< DiffRender<MODELS::TRIANGLE3D_TEXTURED> >();
    impl->init(a_mesh, a_samplesPerPixel);
    return impl;
  }

  return nullptr;
}
