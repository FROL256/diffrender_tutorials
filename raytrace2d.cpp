#include "raytrace.h"
using LiteMath::dot;
using LiteMath::to_float4;
using LiteMath::float4x4;

static inline float VS_X(float V[3], const CamInfo& data)
{
  const float W    = V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = V[0]/W;
  return (xNDC*0.5f + 0.5f)*data.width - 0.5f;
}

static inline float VS_Y(float V[3], const CamInfo& data)
{
  const float W    = V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = -V[1]/W;
  return (xNDC*0.5f + 0.5f)*data.height - 0.5f;
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
    if(m_pMesh->m_geomType == TRIANGLE_3D)
    {
      m_mesh2D = *m_pMesh;
      float4x4 mProj; 
      memcpy(mProj.m_col, cam.projM, 16*sizeof(float));
      for(auto& v : m_mesh2D.vertices) {
        float4 vNDC = mProj*to_float4(v, 1.0f);
        vNDC /= vNDC.w;
        v.x = (vNDC.x*0.5f + 0.5f)*cam.width  - 0.5f;
        v.y = (-vNDC.y*0.5f + 0.5f)*cam.height - 0.5f;
        v.z = vNDC.z;
      }
      //for(auto& v : m_mesh2D.vertices) {
      //  v.x = VS_X(v.M, cam);
      //  v.y = VS_Y(v.M, cam);
      //}
      m_mesh2D.m_geomType = TRIANGLE_2D;
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
      auto A = m_pMesh2D->indices[i+0];
      auto B = m_pMesh2D->indices[i+1];
      auto C = m_pMesh2D->indices[i+2];
      
      auto v0_3d = m_pMesh2D->vertices[A];
      auto v1_3d = m_pMesh2D->vertices[B];
      auto v2_3d = m_pMesh2D->vertices[C];
  
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
  
        const float z = u*v0_3d.z + v*v1_3d.z + (1.0f-u-v)*v2_3d.z;
        
        if(z > hit.t)
        {
          hit.u = u;
          hit.v = v;
          hit.t = z;
        }
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
