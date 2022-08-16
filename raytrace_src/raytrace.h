#pragma once

#include "dmesh.h"
#include <memory>

// /**
// \brief API to ray-scene intersection on CPU
// */
// struct CRT_Hit 
// {
//   float    t;         ///< intersection distance from ray origin to object
//   uint32_t primId; 
//   uint32_t instId;
//   uint32_t geomId;    ///< use 4 most significant bits for geometry type; thay are zero for triangles 
//   float    coords[4]; ///< custom intersection data; for triangles coords[0] and coords[1] stores baricentric coords (u,v)
// };

struct SurfaceInfo
{
  float    t;       ///<! dist from origin ray to surface
  unsigned faceId;  ///<! primitrive id
  float    u;       ///<! first triangle baricentric 
  float    v;       ///<! second triangle baricentric 
};

struct CamInfo
{
  float projM[16];
//float worldViewM[16];
  float width;
  float height;
};

struct IRayTracer
{
  IRayTracer(){}
  virtual ~IRayTracer(){}

  virtual void        Init(const TriangleMesh* pMesh) = 0;
  virtual void        SetCamera(const CamInfo& cam)   = 0;
  virtual SurfaceInfo CastSingleRay(float x, float y, float3* outPos = nullptr, float3* outDir = nullptr) = 0;
};

std::shared_ptr<IRayTracer> MakeRayTracer2D(const char* className);
std::shared_ptr<IRayTracer> MakeRayTracer3D(const char* className);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float3 EyeRayDirNormalized(float x, float y, LiteMath::float4x4 a_mViewProjInv)
{
  float4 pos = float4(2.0f*x - 1.0f, -2.0f*y + 1.0f, 0.0f, 1.0f );
  pos = a_mViewProjInv * pos;
  pos /= pos.w;
  return normalize(to_float3(pos));
}
