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

    auto geomId = m_pAccelStruct->AddGeom_Triangles3f((const float*)m_pMesh->vertices.data(), m_pMesh->vertices.size(), m_pMesh->indices.data(), m_pMesh->indices.size(), BUILD_MEDIUM); 

    m_pAccelStruct->ClearScene();
    m_pAccelStruct->AddInstance(geomId, LiteMath::float4x4()); // with identity matrix
    m_pAccelStruct->CommitScene(BUILD_MEDIUM);
    //std::cout << "[EmbreeRT3D]: Init done" << std::endl;
  }

  void SetCamera(const CamInfo& cam) override
  {
    m_ProjInv      = inverse4x4(cam.mProj);
    m_worldViewInv = inverse4x4(cam.mWorldView);
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
  
    float3 ray_pos = float3(0,0,0);
    float3 ray_dir = EyeRayDirNormalized(x/m_fwidth, y/m_fheight, m_ProjInv);
    float  tNear   = 0.0f;

    transform_ray3f(m_worldViewInv, &ray_pos, &ray_dir);

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

  float4x4 m_ProjInv;
  float4x4 m_worldViewInv;
  float m_fwidth, m_fheight;

};

std::shared_ptr<IRayTracer> MakeEmbreeRT3D() { return std::make_shared<EmbreeRT3D>(); }
