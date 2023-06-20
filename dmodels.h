#pragma once

#include "LiteMath.h"
#include "dmesh.h"
#include "functions.h"
#include <cstdio>

#define DEBUG_RENDER 0
constexpr static int  G_DEBUG_VERT_ID = 0;

struct AuxData
{
  const CamInfo* pCamInfo = nullptr;
  Img* debugImages  = nullptr;
  int debugImageNum = 0;
};

namespace MODELS
{ 
  struct TRIANGLE3D_FACE_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_FACE_COL; }
    
    static inline float3 shade (const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir)
    {
      if (surfInfo.faceId == unsigned(-1))
        return float3(0,0,0); // BGCOLOR
      return mesh.colors[surfInfo.faceId]; 
    }

    static inline void   shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux, 
                                    DTriangleMesh& grad) 
    { 
      GradReal* d_colors = grad.colors_s();

      d_colors[surfElem.faceId*3+0] += GradReal(val.x); 
      d_colors[surfElem.faceId*3+1] += GradReal(val.y);
      d_colors[surfElem.faceId*3+2] += GradReal(val.z);
    }

    static inline void   edge_grad (const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux, 
                                    std::vector<GradReal>& d_pos) 
    { 
      float3 v0_d[2] = {{0,0,0},{0,0,0}}; 
      float3 v1_d[2] = {{0,0,0},{0,0,0}}; 
      
      float3 v0_3d = mesh.vertices[v0];
      float3 v1_3d = mesh.vertices[v1];
      
      VS_X_grad(v0_3d.M, *(aux.pCamInfo), v0_d[0].M);
      VS_Y_grad(v0_3d.M, *(aux.pCamInfo), v0_d[1].M);
      VS_X_grad(v1_3d.M, *(aux.pCamInfo), v1_d[0].M);
      VS_Y_grad(v1_3d.M, *(aux.pCamInfo), v1_d[1].M);
      
      const float dv0_dx = v0_d[0].x*d_v0.x; // + v0_dx.y*d_v0.y; ==> 0
      const float dv0_dy = v0_d[1].y*d_v0.y; // + v0_dy.x*d_v0.x; ==> 0
      const float dv0_dz = (v0_d[0].z*d_v0.x + v0_d[1].z*d_v0.y); 
       
      const float dv1_dx = v1_d[0].x*d_v1.x; // + v1_dx.y*d_v1.y; ==> 0
      const float dv1_dy = v1_d[1].y*d_v1.y; // + v1_dy.x*d_v1.x; ==> 0
      const float dv1_dz = (v1_d[0].z*d_v1.x + v1_d[1].z*d_v1.y); 
      
      //#if DEBUG_RENDER
      //for(int debugId=0; debugId<3; debugId++) 
      //{
      //  if(G_DEBUG_VERT_ID + debugId == v0)
      //  {
      //    if(aux.debugImageNum > 0 && aux.debugImages!= nullptr)
      //      aux.debugImages[debugId][int2(xi,yi)] += float3(dv0_dx,dv0_dy,dv0_dz);
      //  }
      //  else if(G_DEBUG_VERT_ID + debugId == v1)
      //  {
      //    if(aux.debugImageNum > 0)
      //      aux.debugImages[debugId][int2(xi,yi)] += float3(dv1_dx,dv1_dy,dv1_dz);
      //  }
      //}
      //#endif
      
      d_pos[v0*3+0] += GradReal(dv0_dx);
      d_pos[v0*3+1] += GradReal(dv0_dy);
      d_pos[v0*3+2] += GradReal(dv0_dz);
      
      d_pos[v1*3+0] += GradReal(dv1_dx);
      d_pos[v1*3+1] += GradReal(dv1_dy);
      d_pos[v1*3+2] += GradReal(dv1_dz);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
  struct TRIANGLE3D_VERT_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_VERT_COL; }
  
    static inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir)
    { 
      if (surfInfo.faceId == unsigned(-1))
        return float3(0,0,0); // BGCOLOR

      const auto  A = mesh.indices[surfInfo.faceId*3+0];
      const auto  B = mesh.indices[surfInfo.faceId*3+1];
      const auto  C = mesh.indices[surfInfo.faceId*3+2];
      const float u = surfInfo.u;
      const float v = surfInfo.v;
      return mesh.colors[A]*(1.0f-u-v) + mesh.colors[B]*v + u*mesh.colors[C]; 
    }
  
    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux, 
                                  DTriangleMesh& grad)
    {
      GradReal* d_colors = grad.colors_s();
      GradReal* d_pos    = grad.vertices_s();
  
      auto A = mesh.indices[surfElem.faceId*3+0];
      auto B = mesh.indices[surfElem.faceId*3+1];
      auto C = mesh.indices[surfElem.faceId*3+2];
      
      auto contribA = (1.0f-surfElem.u-surfElem.v)*val;
      auto contribB = surfElem.v*val;
      auto contribC = surfElem.u*val;
            
      d_colors[A*3+0] += GradReal(contribA.x);
      d_colors[A*3+1] += GradReal(contribA.y);
      d_colors[A*3+2] += GradReal(contribA.z);
      
      d_colors[B*3+0] += GradReal(contribB.x);
      d_colors[B*3+1] += GradReal(contribB.y);
      d_colors[B*3+2] += GradReal(contribB.z);
      
      d_colors[C*3+0] += GradReal(contribC.x);
      d_colors[C*3+1] += GradReal(contribC.y);
      d_colors[C*3+2] += GradReal(contribC.z);
        
      const float3 c0 = mesh.colors[A];
      const float3 c1 = mesh.colors[B];
      const float3 c2 = mesh.colors[C];  
      const float dF_dU = dot((c2-c0), val);
      const float dF_dV = dot((c1-c0), val);
            
      if(dF_dU*dF_dU > 0.0f || dF_dV*dF_dV > 0.0f) 
      {
        const float3 v0 = mesh.vertices[A];
        const float3 v1 = mesh.vertices[B];
        const float3 v2 = mesh.vertices[C];
        float3 dU_dvert[3] = {};
        float3 dV_dvert[3] = {};
        
        BarU_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, dU_dvert[0].M, dU_dvert[1].M, dU_dvert[2].M);
        BarV_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, dV_dvert[0].M, dV_dvert[1].M, dV_dvert[2].M);
      
        auto contribVA = (dF_dU*dU_dvert[0] + dF_dV*dV_dvert[0]);  
        auto contribVB = (dF_dU*dU_dvert[1] + dF_dV*dV_dvert[1]);  
        auto contribVC = (dF_dU*dU_dvert[2] + dF_dV*dV_dvert[2]);  
        
        //#if DEBUG_RENDER
        //for(int debugId=0; debugId<3; debugId++) 
        //{
        //  if(G_DEBUG_VERT_ID+debugId == A || G_DEBUG_VERT_ID+debugId == B || G_DEBUG_VERT_ID+debugId == C)
        //  {
        //    auto contrib = contribVA;
        //    if(G_DEBUG_VERT_ID+debugId == B)
        //      contrib = contribVB;
        //    else if(G_DEBUG_VERT_ID+debugId == C)
        //      contrib = contribVC;
        //    //contrib *= float3(0.1f, 0.1f, 1.0f);
        //    if(aux.debugImageNum > debugId && aux.debugImages!= nullptr)
        //      aux.debugImages[debugId][int2(x,y)] += contrib;
        //  }
        //}
        //#endif

        d_pos[A*3+0] += GradReal(contribVA.x);
        d_pos[A*3+1] += GradReal(contribVA.y);
        d_pos[A*3+2] += GradReal(contribVA.z);
        
        d_pos[B*3+0] += GradReal(contribVB.x);
        d_pos[B*3+1] += GradReal(contribVB.y);
        d_pos[B*3+2] += GradReal(contribVB.z);
        
        d_pos[C*3+0] += GradReal(contribVC.x);
        d_pos[C*3+1] += GradReal(contribVC.y);
        d_pos[C*3+2] += GradReal(contribVC.z);
      }
    }

    static inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux, 
                                 std::vector<GradReal>& d_pos) 
    {
      TRIANGLE3D_FACE_COLOR::edge_grad(mesh, v0, v1, d_v0, d_v1, aux, 
                                       d_pos);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct TRIANGLE3D_TEXTURED
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_DIFF_TEX; }
  
    static inline std::vector<float> sample_bilinear_clamp(float2 tc, const CPUTexture &tex)
    {
      tc *= float2(tex.w, tex.h);
      int2 tc0 = clamp(int2(tc), int2(0,0), int2(tex.w-1, tex.h-1));
      int2 tc1 = clamp(int2(tc) + int2(1,1), int2(0,0), int2(tex.w-1, tex.h-1));
      float2 dtc = tc - float2(tc0);
      const float *p00 = tex.get(tc0.x, tc0.y);
      const float *p01 = tex.get(tc0.x, tc1.y);
      const float *p10 = tex.get(tc1.x, tc0.y);
      const float *p11 = tex.get(tc1.x, tc1.y);

      std::vector<float> res(tex.channels, 0);
      for (int i=0;i<tex.channels;i++)
      {
        res[i] = (1-dtc.x)*(1-dtc.y)*p00[i] + (1-dtc.x)*dtc.y*p01[i] + dtc.x*(1-dtc.y)*p10[i] + dtc.x*dtc.y*p11[i];
      }

      return res;
    }

    static inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir)
    {
      if (surfInfo.faceId == unsigned(-1))
        return float3(0,0,0); // BGCOLOR

      const auto  A = mesh.indices[surfInfo.faceId*3+0];
      const auto  B = mesh.indices[surfInfo.faceId*3+1];
      const auto  C = mesh.indices[surfInfo.faceId*3+2];
      const float u = surfInfo.u;
      const float v = surfInfo.v;
      float2 tc = mesh.tc[A]*(1.0f-u-v) + mesh.tc[B]*v + u*mesh.tc[C];

      switch (mesh.material)
      {
      case MATERIAL::DIFFUSE:
        {
          assert(mesh.textures.size()>=1 && mesh.textures[0].channels == 3);
          auto res = sample_bilinear_clamp(tc, mesh.textures[0]);
          return float3(res[0],res[1],res[2]);
        }
        break;
      
      default:
        return float3(0,0,0); // BGCOLOR
        break;
      } 
    }
  
    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux, 
                                  DTriangleMesh& grad)
    {
      auto A = mesh.indices[surfElem.faceId*3+0];
      auto B = mesh.indices[surfElem.faceId*3+1];
      auto C = mesh.indices[surfElem.faceId*3+2];
      
      const float u = surfElem.u;
      const float v = surfElem.v;
      float2 tc = mesh.tc[A]*(1.0f-u-v) + mesh.tc[B]*v + u*mesh.tc[C];

      switch (mesh.material)
      {
      case MATERIAL::DIFFUSE:
        {
          assert(mesh.textures.size()>=1 && mesh.textures[0].channels == 3);
          auto &tex = mesh.textures[0];
          
          tc *= float2(tex.w, tex.h);
          int2 tc0 = clamp(int2(tc), int2(0,0), int2(tex.w-1, tex.h-1));
          int2 tc1 = clamp(int2(tc) + int2(1,1), int2(0,0), int2(tex.w-1, tex.h-1));
          float2 dtc = tc - float2(tc0);
          int off = grad.tex_offset(0);

          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y)] += (1-dtc.x)*(1-dtc.y)*val.x;
          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y)+1] += (1-dtc.x)*(1-dtc.y)*val.y;
          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc0.y)+2] += (1-dtc.x)*(1-dtc.y)*val.z;

          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y)] += (1-dtc.x)*dtc.y*val.x;
          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y)+1] += (1-dtc.x)*dtc.y*val.y;
          grad[off + mesh.textures[0].pixel_to_offset(tc0.x, tc1.y)+2] += (1-dtc.x)*dtc.y*val.z;

          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y)] += dtc.x*(1-dtc.y)*val.x;
          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y)+1] += dtc.x*(1-dtc.y)*val.y;
          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc0.y)+2] += dtc.x*(1-dtc.y)*val.z;

          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y)] += dtc.x*dtc.y*val.x;
          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y)+1] += dtc.x*dtc.y*val.y;
          grad[off + mesh.textures[0].pixel_to_offset(tc1.x, tc1.y)+2] += dtc.x*dtc.y*val.z;
        }
        break;
      
      default:
        break;
      }             
    }

    static inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux, 
                                 std::vector<GradReal>& d_pos) 
    {
      TRIANGLE3D_FACE_COLOR::edge_grad(mesh, v0, v1, d_v0, d_v1, aux, 
                                       d_pos);
    }
  };
};