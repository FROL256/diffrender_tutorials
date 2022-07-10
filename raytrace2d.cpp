#include "raytrace.h"
using LiteMath::dot;

struct BruteForce2D : public IRayTracer
{
  BruteForce2D(){}
  ~BruteForce2D() override {}

  void Init(const TriangleMesh* pMesh) override 
  {
    m_pMesh = pMesh;
  }

  SurfaceInfo CastSingleRay(float x, float y) override
  {
    SurfaceInfo hit;
    hit.faceId = unsigned(-1);
    hit.u      = 0.0f;
    hit.v      = 0.0f;

    const TriangleMesh& mesh = *m_pMesh;
    const float2 screen_pos(x,y);
  
    // loop over all triangles in a mesh, return the first one that hits
    for (size_t i = 0; i < (int)mesh.indices.size(); i+=3) 
    {
      // retrieve the three vertices of a triangle
      auto A = mesh.indices[i+0];
      auto B = mesh.indices[i+1];
      auto C = mesh.indices[i+2];
      
      auto v0_3d = mesh.vertices[A];
      auto v1_3d = mesh.vertices[B];
      auto v2_3d = mesh.vertices[C];
  
      float2 v0 = float2(v0_3d.x, v0_3d.y);
      float2 v1 = float2(v1_3d.x, v1_3d.y);
      float2 v2 = float2(v2_3d.x, v2_3d.y);
  
      // form three half-planes: v1-v0, v2-v1, v0-v2
      // if a point is on the same side of all three half-planes, it's inside the triangle.
      auto n01 = normal2D(v1 - v0);
      auto n12 = normal2D(v2 - v1);
      auto n20 = normal2D(v0 - v2);
      
      const bool side01 = dot(screen_pos - v0, n01) > 0;
      const bool side12 = dot(screen_pos - v1, n12) > 0;
      const bool side20 = dot(screen_pos - v2, n20) > 0;
      if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) 
      { 
        hit.faceId = (i/3);
  
        const float areaInv = 1.0f / edgeFunction(v0, v1, v2); 
        const float e0      = edgeFunction(v0, v1, screen_pos);
        const float e1      = edgeFunction(v1, v2, screen_pos);
        const float e2      = edgeFunction(v2, v0, screen_pos);
        const float u = e1*areaInv; // v0
        const float v = e2*areaInv; // v1 
  
        hit.u = u;
        hit.v = v;
        break;
      }
    }
  
    return hit;
  }

  const TriangleMesh* m_pMesh = nullptr;

};

std::shared_ptr<IRayTracer> MakeRayTracer2D(const char* className)
{
  return std::make_shared<BruteForce2D>();
}
