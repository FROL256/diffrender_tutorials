#pragma once

#include "LiteMath.h"
#include "dmesh.h"
#include "raytrace.h"
#include "functions.h"

namespace MODELS
{ 
  struct TRIANGLE2D_FACE_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_FACE_COL; }
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_2D; }
    
    /**
     \brief eval shading: BSDF, lighting, colors and e.t.c
     \param mesh     -- mesh
     \param surfInfo -- current surface point
     \param ray_pos  -- input ray (from camera tu surface) origin
     \param ray_dir  -- input ray (from camera tu surface) direction
    */
    static inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir)
    {
      if (surfInfo.faceId == unsigned(-1))
        return float3(0,0,0); // BGCOLOR
      return mesh.colors[surfInfo.faceId]; 
    }
    
    /**
     \brief gradient of shade function that will be used for interior derivatives
     \param mesh     -- mesh
     \param surfInfo -- current surface point
     \param ray_pos  -- input ray (from camera tu surface) origin
     \param ray_dir  -- input ray (from camera tu surface) direction
     \param val      -- input error value that we need to backpropagate further to ,esh
     \param grad     -- output gradient
    */
    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val,
                                  DTriangleMesh& grad)
    {
      GradReal* d_colors = grad.colors_s();
      //GradReal* d_pos    = grad.vertices_s();
      d_colors[surfElem.faceId*3+0] += GradReal(val.x); 
      d_colors[surfElem.faceId*3+1] += GradReal(val.y);
      d_colors[surfElem.faceId*3+2] += GradReal(val.z);
    }
    
    /**
     \brief gradient of shade discontinuity
     \param v0       -- first  vertex to contribute
     \param v1       -- second vertex to contribute
     \param d_v0     -- (dF/dv0.x, dF/dv0.y) in 2D space 
     \param d_v1     -- (dF/dv1.x, dF/dv1.y) in 2D space 
     \param ray_dir  -- input ray (from camera tu surface) direction
     \param val      -- input error value that we need to backpropagate further to ,esh
     \param grad     -- output gradient
    */
    static inline void edge_grad(const int v0, const int v1, const float2 d_v0, const float2 d_v1,
                                 DTriangleMesh& grad)
    {
      GradReal* d_pos = grad.vertices_s();

      d_pos[v0*3+0] += GradReal(d_v0.x);
      d_pos[v0*3+1] += GradReal(d_v0.y);
      
      d_pos[v1*3+0] += GradReal(d_v1.x);
      d_pos[v1*3+1] += GradReal(d_v1.y);
    }

  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct TRIANGLE2D_VERT_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_VERT_COL; }
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_2D; }

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

    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val,
                                  DTriangleMesh& grad)
    {
      GradReal* d_colors = grad.colors_s();
      //GradReal* d_pos    = grad.vertices_s();

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
    }

    static inline void edge_grad(const int v0, const int v1, const float2 d_v0, const float2 d_v1, DTriangleMesh& grad) { TRIANGLE2D_FACE_COLOR::edge_grad(v0,v1, d_v0, d_v1, grad); }
  };
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct TRIANGLE3D_FACE_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_FACE_COL; }
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_3D; }
    
    static inline float3 shade     (const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir) { return TRIANGLE2D_FACE_COLOR::shade(mesh, surfInfo, ray_pos, ray_dir); }
    static inline void   shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, DTriangleMesh& grad) { TRIANGLE2D_FACE_COLOR::shade_grad(mesh, surfElem, ray_pos, ray_dir, val, grad); }
    static inline void   edge_grad (const int v0, const int v1, const float2 d_v0, const float2 d_v1, DTriangleMesh& grad) 
    {
      //GradReal* d_pos = grad.vertices_s();
      //
      //float3 v0_d[2] = {{0,0,0},{0,0,0}}; 
      //float3 v1_d[2] = {{0,0,0},{0,0,0}}; 
      //
      //float3 v0_3d = mesh3d.vertices[v0];
      //float3 v1_3d = mesh3d.vertices[v1];
      //VS_X_grad(v0_3d.M, g_uniforms, v0_d[0].M);
      //VS_Y_grad(v0_3d.M, g_uniforms, v0_d[1].M);
      //VS_X_grad(v1_3d.M, g_uniforms, v1_d[0].M);
      //VS_Y_grad(v1_3d.M, g_uniforms, v1_d[1].M);
      //
      //const float dv0_dx = v0_d[0].x*d_v0.x; //  + v0_dx.y*d_v0.y;
      //const float dv0_dy = v0_d[1].y*d_v0.y; //  + v0_dy.x*d_v0.x;
      //const float dv0_dz = (v0_d[0].z*d_v0.x + v0_d[1].z*d_v0.y); 
      // 
      //const float dv1_dx = v1_d[0].x*d_v1.x; // + v1_dx.y*d_v1.y;
      //const float dv1_dy = v1_d[1].y*d_v1.y; // + v1_dy.x*d_v1.x;
      //const float dv1_dz = (v1_d[0].z*d_v1.x + v1_d[1].z*d_v1.y); 
      //
      //#if DEBUG_RENDER
      //for(int debugId=0; debugId<3; debugId++) 
      //{
      //  if(G_DEBUG_VERT_ID + debugId == edge.v0)
      //  {
      //    if(debugImageNum > 0 && debugImages!= nullptr)
      //      debugImages[debugId][int2(xi,yi)] += float3(dv0_dx,dv0_dy,dv0_dz);
      //  }
      //  else if(G_DEBUG_VERT_ID + debugId == edge.v1)
      //  {
      //    if(debugImageNum > 0)
      //      debugImages[debugId][int2(xi,yi)] += float3(dv1_dx,dv1_dy,dv1_dz);
      //  }
      //}
      //#endif
      //
      //d_pos[v0*3+0] += GradReal(dv0_dx);
      //d_pos[v0*3+1] += GradReal(dv0_dy);
      //d_pos[v0*3+2] += GradReal(dv0_dz);
      //
      //d_pos[v1*3+0] += GradReal(dv1_dx);
      //d_pos[v1*3+1] += GradReal(dv1_dy);
      //d_pos[v1*3+2] += GradReal(dv1_dz);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
  struct TRIANGLE3D_VERT_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_VERT_COL; }
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_3D; }
  
    static inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir) { return TRIANGLE2D_VERT_COLOR::shade(mesh, surfInfo, ray_pos, ray_dir); }
  
    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val,
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
        
        #if DEBUG_RENDER
        for(int debugId=0; debugId<3; debugId++) 
        {
          if(G_DEBUG_VERT_ID+debugId == A || G_DEBUG_VERT_ID+debugId == B || G_DEBUG_VERT_ID+debugId == C)
          {
            auto contrib = contribVA;
            if(G_DEBUG_VERT_ID+debugId == B)
              contrib = contribVB;
            else if(G_DEBUG_VERT_ID+debugId == C)
              contrib = contribVC;
            //contrib *= float3(0.1f, 0.1f, 1.0f);
            if(debugImageNum > debugId && debugImages!= nullptr)
              debugImages[debugId][int2(x,y)] += contrib;
          }
        }
        #endif

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

    static inline void edge_grad(const int v0, const int v1, const float2 d_v0, const float2 d_v1, DTriangleMesh& grad) 
    {
      TRIANGLE3D_FACE_COLOR::edge_grad(v0, v1, d_v0, d_v1, grad);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
};

template<class MeshType, class GeomType>
struct DiffRender
{
  
};

