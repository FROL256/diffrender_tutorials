#include "raytrace.h"
using LiteMath::dot;
using LiteMath::sign;
using LiteMath::cross;
using LiteMath::float4x4;
using LiteMath::float3;
using LiteMath::normalize;
using LiteMath::inverse4x4;
using LiteMath::to_float3;

static inline float3 EyeRayDirNormalized(float x, float y, float4x4 a_mViewProjInv)
{
  float4 pos = float4(2.0f*x - 1.0f, -2.0f*y + 1.0f, 0.0f, 1.0f );
  pos = a_mViewProjInv * pos;
  pos /= pos.w;
  return normalize(to_float3(pos));
}

struct BruteForce3D : public IRayTracer
{
  BruteForce3D(){}
  ~BruteForce3D() override {}

  void Init(const TriangleMesh* pMesh) override 
  {
    m_pMesh = pMesh;
  }

  void SetCamera(const CamInfo& cam) override
  {
    memcpy(m_mViewProjInv.m_col, cam.projM, 16*sizeof(float));
    m_mViewProjInv = inverse4x4(m_mViewProjInv);
    m_fwidth       = cam.width;
    m_fheight      = cam.height;
  }

  SurfaceInfo CastSingleRay(float x, float y) override
  {
    SurfaceInfo hit;
    hit.faceId = unsigned(-1);
    hit.u      = 0.0f;
    hit.v      = 0.0f;
    hit.t      = 1e+6f; // tFar

    const TriangleMesh& mesh = *m_pMesh;
    const float2 screen_pos(x,y);
  
    const float3 ray_pos = float3(0,0,0);
    const float3 ray_dir = EyeRayDirNormalized(x/m_fwidth, y/m_fheight, m_mViewProjInv);
    const float  tNear   = 0.0f;

    for (size_t triAddress = 0; triAddress < m_pMesh->indices.size(); triAddress += 3)
    { 
      const uint A = m_pMesh->indices[triAddress + 0];
      const uint B = m_pMesh->indices[triAddress + 1];
      const uint C = m_pMesh->indices[triAddress + 2];
    
      const float3 A_pos = m_pMesh->vertices[A];
      const float3 B_pos = m_pMesh->vertices[B];
      const float3 C_pos = m_pMesh->vertices[C];
    
      const float3 edge1 = B_pos - A_pos;
      const float3 edge2 = C_pos - A_pos;
      const float3 pvec  = cross(ray_dir, edge2);
      const float3 tvec  = ray_pos - A_pos;
      const float3 qvec  = cross(tvec, edge1);
      const float  e1dpv = dot(edge1, pvec);
      const float  signv = sign(e1dpv);                 // put 1.0 to enable triangle clipping
      const float invDet = signv / std::max(signv*e1dpv, 1e-6f);
    
      const float v = dot(tvec, pvec)*invDet;
      const float u = dot(qvec, ray_dir)*invDet;
      const float t = dot(edge2, qvec)*invDet;
    
      if (v > 0.0f && u > 0.0f && (u + v < 1.0f) && t > tNear && t < hit.t)
      {
        hit.t      = t;
        hit.faceId = triAddress/3;
        hit.u      = 1.0f-u-v;    // v0
        hit.v      = v;           // v1
      }
    }
  
    return hit;
  }

  const TriangleMesh* m_pMesh = nullptr;
  float4x4 m_mViewProjInv;
  float m_fwidth, m_fheight;

};

std::shared_ptr<IRayTracer> MakeRayTracer3D(const char* className)
{
  return std::make_shared<BruteForce3D>();
}
