#include <cstdint>
#include <memory>
//#include <iostream>
#include <vector>

#include "raytrace.h"
#include "LiteMath.h"
#include "CrossRT.h"

using LiteMath::dot;
using LiteMath::sign;
using LiteMath::cross;
using LiteMath::float4x4;
using LiteMath::float3;
using LiteMath::normalize;
using LiteMath::inverse4x4;
using LiteMath::to_float3;

struct EmbreeRT3D : public IRayTracer
{
  EmbreeRT3D(){}
  ~EmbreeRT3D() override {}

  void Init(const TriangleMesh* pMesh) override 
  {
    m_pMesh = pMesh;

    m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT(""));
    m_pAccelStruct->ClearGeom();

    std::vector<float4> vertCopy(m_pMesh->vertices.size());
    for(size_t i=0;i<vertCopy.size();i++)
      vertCopy[i] = to_float4(m_pMesh->vertices[i], 1.0f);

    auto geomId = m_pAccelStruct->AddGeom_Triangles4f(vertCopy.data(), vertCopy.size(), m_pMesh->indices.data(), m_pMesh->indices.size()); // TODO: AddGeom_Triangles3f also, we need this

    m_pAccelStruct->ClearScene();
    m_pAccelStruct->AddInstance(geomId, LiteMath::float4x4()); // with identity matrix
    m_pAccelStruct->CommitScene();
    //std::cout << "[EmbreeRT3D]: Init done" << std::endl;
  }

  void SetCamera(const CamInfo& cam) override
  {
    memcpy(m_mViewProjInv.m_col, cam.projM, 16*sizeof(float));
    m_mViewProjInv = inverse4x4(m_mViewProjInv);
    m_fwidth       = cam.width;
    m_fheight      = cam.height;
  }

  SurfaceInfo CastSingleRay(float x, float y, float3* outPos, float3* outDir) override
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

    if(outPos != nullptr)
      *outPos = ray_pos;
    if(outDir != nullptr)
      *outDir = ray_dir;

    CRT_Hit crtHit = m_pAccelStruct->RayQuery_NearestHit(to_float4(ray_pos, 0.0f), to_float4(ray_dir, 1e10f));

    hit.faceId = crtHit.primId;
    hit.t      = crtHit.t;
    hit.u      = crtHit.coords[0];
    hit.v      = crtHit.coords[1];
    return hit;
  }

  const TriangleMesh* m_pMesh = nullptr;
  std::shared_ptr<ISceneObject> m_pAccelStruct;

  float4x4 m_mViewProjInv;
  float m_fwidth, m_fheight;

};

std::shared_ptr<IRayTracer> MakeEmbreeRT3D() { return std::make_shared<EmbreeRT3D>(); }
