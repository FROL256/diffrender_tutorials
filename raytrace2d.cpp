#include "raytrace.h"
using LiteMath::dot;
using LiteMath::to_float4;
using LiteMath::float4x4;

static inline float VS_X(float V[3], const CamInfo& data)
{
  const float W    =  V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = (V[0] * data.projM[0] + V[1] * data.projM[4] + V[2] * data.projM[ 8] + data.projM[12])/W;
  return (xNDC*0.5f + 0.5f)*data.width;
}

static inline float VS_Y(float V[3], const CamInfo& data)
{
  const float W    =   V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = -(V[0] * data.projM[1] + V[1] * data.projM[5] + V[2] * data.projM[ 9] + data.projM[13])/W;
  return (xNDC*0.5f + 0.5f)*data.height;
}

struct BruteForce2D : public IRayTracer
{
  BruteForce2D(){}
  ~BruteForce2D() override {}

  void Init(const TriangleMesh* pMesh) override 
  {
    m_pMesh = pMesh;
    m_pMesh2D = nullptr;
  }

  void SetCamera(const CamInfo& cam) override
  {
    if(m_pMesh->m_geomType == GEOM_TYPE::TRIANGLE_3D)
    {
      m_mesh2D = *m_pMesh;
      for(auto& v : m_mesh2D.vertices) {
        auto vCopy = v;
        v.x = VS_X(vCopy.M, cam);
        v.y = VS_Y(vCopy.M, cam);
      }
      m_mesh2D.m_geomType = GEOM_TYPE::TRIANGLE_2D;
      m_pMesh2D           = &m_mesh2D;
    }
    else
      m_pMesh2D = m_pMesh;
  }

  SurfaceInfo CastSingleRay(float x, float y) override
  {
    SurfaceInfo hit;
    hit.faceId = unsigned(-1);
    hit.u      = 0.0f;
    hit.v      = 0.0f;
    hit.t      = 0.0f;

    const float2 screen_pos(x,y);
  
    // loop over all triangles in a mesh, return the first one that hits
    for (size_t i = 0; i < (int)m_pMesh2D->indices.size(); i+=3) 
    {
      // retrieve the three vertices of a triangle
      const auto A = m_pMesh2D->indices[i+0];
      const auto B = m_pMesh2D->indices[i+1];
      const auto C = m_pMesh2D->indices[i+2];
      
      const float2 v0 = LiteMath::to_float2(m_pMesh2D->vertices[A]);
      const float2 v1 = LiteMath::to_float2(m_pMesh2D->vertices[B]);
      const float2 v2 = LiteMath::to_float2(m_pMesh2D->vertices[C]);
  
      // form three half-planes: v1-v0, v2-v1, v0-v2
      // if a point is on the same side of all three half-planes, it's inside the triangle.
      const auto n01 = normal2D(v1 - v0);
      const auto n12 = normal2D(v2 - v1);
      const auto n20 = normal2D(v0 - v2);
      
      const bool side01 = dot(screen_pos - v0, n01) > 0;
      const bool side12 = dot(screen_pos - v1, n12) > 0;
      const bool side20 = dot(screen_pos - v2, n20) > 0;
      if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) 
      { 
        hit.faceId = (i/3);
  
        const float areaInv = 1.0f / edgeFunction(v0, v1, v2); 
        //const float e0      = edgeFunction(v0, v1, screen_pos);
        const float e1      = edgeFunction(v1, v2, screen_pos);
        const float e2      = edgeFunction(v2, v0, screen_pos);
        hit.u = e1*areaInv; // v0
        hit.v = e2*areaInv; // v1 
      }
    }
  
    return hit;
  }

  const TriangleMesh* m_pMesh   = nullptr;
  const TriangleMesh* m_pMesh2D = nullptr;
  TriangleMesh m_mesh2D;

};

std::shared_ptr<IRayTracer> MakeRayTracer2D(const char* className)
{
  return std::make_shared<BruteForce2D>();
}
