#pragma once

#include "LiteMath.h"
#include "dmesh.h"
#include "raytrace.h"
#include "functions.h"

#define DEBUG_RENDER 0

#if DEBUG_RENDER
constexpr static int  MAXTHREADS    = 1;
#else
constexpr static int  MAXTHREADS    = 14;
#endif

struct AuxData
{
  CamInfo* pCamInfo = nullptr;
  Img* debugImages  = nullptr;
  int debugImageNum = 0;
};

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
     \param val      -- input error value that we need to backpropagate further to mesh
     \param aux      -- auxilarry data (constants, debugging and e.t.c)
     \param grad     -- output gradient
    */
    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux,
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
     \param mesh     -- mesh
     \param v0       -- first  vertex to contribute
     \param v1       -- second vertex to contribute
     \param d_v0     -- (dF/dv0.x, dF/dv0.y) in 2D space 
     \param d_v1     -- (dF/dv1.x, dF/dv1.y) in 2D space 
     \param ray_dir  -- input ray (from camera tu surface) direction
     \param val      -- input error value that we need to backpropagate further to ,esh
     \param grad     -- output gradient
    */
    static inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux,
                                 std::vector<GradReal>& d_pos)
    {
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

    static inline void shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux,
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

    static inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux, 
                                 std::vector<GradReal>& d_pos) 
    { 
      TRIANGLE2D_FACE_COLOR::edge_grad(mesh, v0, v1, d_v0, d_v1, aux, 
                                       d_pos); 
    }
  };
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct TRIANGLE3D_FACE_COLOR
  {
    static inline MESH_TYPES getMeshType() { return MESH_TYPES::TRIANGLE_FACE_COL; }
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_3D; }
    
    static inline float3 shade     (const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir) { return TRIANGLE2D_FACE_COLOR::shade(mesh, surfInfo, ray_pos, ray_dir); }

    static inline void   shade_grad(const TriangleMesh &mesh, const SurfaceInfo& surfElem, const float3 ray_pos, const float3 ray_dir, const float3 val, const AuxData aux, 
                                    DTriangleMesh& grad) 
    { 
        TRIANGLE2D_FACE_COLOR::shade_grad(mesh, surfElem, ray_pos, ray_dir, val, aux, 
                                          grad); 
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
      
      #if DEBUG_RENDER
      for(int debugId=0; debugId<3; debugId++) 
      {
        if(G_DEBUG_VERT_ID + debugId == v0)
        {
          if(aux.debugImageNum > 0 && aux.debugImages!= nullptr)
            aux.debugImages[debugId][int2(xi,yi)] += float3(dv0_dx,dv0_dy,dv0_dz);
        }
        else if(G_DEBUG_VERT_ID + debugId == v1)
        {
          if(aux.debugImageNum > 0)
            aux.debugImages[debugId][int2(xi,yi)] += float3(dv1_dx,dv1_dy,dv1_dz);
        }
      }
      #endif
      
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
    static inline GEOM_TYPES getGeomType() { return GEOM_TYPES::TRIANGLE_3D; }
  
    static inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo, const float3 ray_pos, const float3 ray_dir) { return TRIANGLE2D_VERT_COLOR::shade(mesh, surfInfo, ray_pos, ray_dir); }
  
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
            if(aux.debugImageNum > debugId && aux.debugImages!= nullptr)
              aux.debugImages[debugId][int2(x,y)] += contrib;
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

    static inline void edge_grad(const TriangleMesh &mesh, const int v0, const int v1, const float2 d_v0, const float2 d_v1, const AuxData aux, 
                                 std::vector<GradReal>& d_pos) 
    {
      TRIANGLE3D_FACE_COLOR::edge_grad(mesh, v0, v1, d_v0, d_v1, aux, 
                                       d_pos);
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <omp.h>
#include "qmc.h"
#include <vector>
#include <set>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::int2;

using LiteMath::clamp;
using LiteMath::normalize;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Edge 
{
  int v0, v1; // vertex ID, v0 < v1
  Edge(int a_v0, int a_v1) : v0(std::min(a_v0, a_v1)), v1(std::max(a_v0, a_v1)) {}
  bool operator<(const Edge &e) const { return this->v0 != e.v0 ? this->v0 < e.v0 : this->v1 < e.v1; } // for sorting edges
};

// for sampling edges with inverse transform sampling
struct Sampler 
{
  std::vector<float> pmf; // probability mass function
  std::vector<float> cdf;
};

// build a discrete CDF using edge length
Sampler build_edge_sampler(const TriangleMesh &mesh, const std::vector<Edge> &edges);

// binary search for inverting the CDF in the sampler
int sample(const Sampler &sampler, const float u);

// given a triangle mesh, collect all edges.
std::vector<Edge> collect_edges(const TriangleMesh &mesh);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Model>
struct DiffRender
{
  void init(const TriangleMesh &a_mesh, int a_samplesPerPixel)
  {
    m_samples_per_pixel = a_samplesPerPixel;
    if(a_mesh.m_geomType == GEOM_TYPES::TRIANGLE_2D)
      m_pTracer = MakeRayTracer2D("");  
    else
      m_pTracer = MakeRayTracer3D("");
    
    m_hammSamples.resize(2*a_samplesPerPixel);

    qmc::init(m_table);
    qmc::planeHammersley(m_hammSamples.data(), a_samplesPerPixel);
  }

  void render(const TriangleMesh &mesh, const CamInfo& cam, Img &img) // TODO: add BSPImage rendering
  {
    auto sqrt_num_samples  = (int)sqrt((float)m_samples_per_pixel);
    auto samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    m_pTracer->Init(&mesh);
    m_pTracer->SetCamera(cam);

    #pragma omp parallel for collapse (2) num_threads(MAXTHREADS) 
    for (int y = 0; y < img.height(); y++) { // for each pixel 
      for (int x = 0; x < img.width(); x++) {
        
        float3 pixelColor = float3(0,0,0);

        for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
          for (int dx = 0; dx < sqrt_num_samples; dx++) {

            auto xoff = (dx + 0.5f) / float(sqrt_num_samples);
            auto yoff = (dy + 0.5f) / float(sqrt_num_samples);
            auto screen_pos = float2{x + xoff, y + yoff};
            
            float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
            auto surf  = m_pTracer->CastSingleRay(screen_pos.x, screen_pos.y, &ray_pos, &ray_dir);
            auto color = Model::shade(mesh, surf, ray_pos, ray_dir);

            pixelColor += (color / samples_per_pixel);
          }
        }

        img[int2(x,y)] = pixelColor;
      }
    }
  }

  
  void d_render(const TriangleMesh &mesh, const CamInfo& cam,
                const Img &adjoint,
                const int edge_samples_in_total,
                DTriangleMesh &d_mesh,
                Img* debugImages, int debugImageNum) 
  {  
    // Build Acceleration structurres and e.t.c. if needed
    //
    m_pTracer->Init(&mesh);
    m_pTracer->SetCamera(cam);

    m_aux.pCamInfo      = &cam;
    m_aux.debugImages   = debugImages;
    m_aux.debugImageNum = debugImageNum;
  
    interior_derivatives(mesh, adjoint, 
                         d_mesh);
    
    //edge_derivatives(mesh, adjoint, edge_samples_in_total,
    //                 d_mesh, debugImages, debugImageNum);
  }

private:

  void interior_derivatives(const TriangleMesh &mesh, const Img &adjoint,
                            DTriangleMesh &d_mesh) 
  {
    auto sqrt_num_samples = (int)sqrt((float)m_samples_per_pixel);
    auto samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
  
    DTriangleMesh grads[MAXTHREADS];
    for(int i=0;i<MAXTHREADS;i++) {
      grads[i] = d_mesh;
      //grads[i].clear(); // don't have to do this explicitly because 'compute_interior_derivatives' is called in the first pass and gradient is already 0.0
    }

    #pragma omp parallel for collapse (2) num_threads(MAXTHREADS)
    for (int y = 0; y < adjoint.height(); y++) { // for each pixel  
      for (int x = 0; x < adjoint.width(); x++)  {
  
        for (int samId = 0; samId < samples_per_pixel; samId++) // for each subpixel
        {         
          float xoff = m_hammSamples[2*samId+0];
          float yoff = m_hammSamples[2*samId+1];
          
          float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
          auto surfElem  = m_pTracer->CastSingleRay(x + xoff, y + yoff, &ray_pos, &ray_dir);
  
          if (surfElem.faceId != unsigned(-1)) {          
            const auto val = adjoint[int2(x,y)] / samples_per_pixel;
            Model::shade_grad(mesh, surfElem, ray_pos, ray_dir, val, m_aux, 
                              grads[omp_get_thread_num()]);
          }      
               
        } // for (int samId = 0; samId < samples_per_pixel; samId++)
      }
    }
  
    // accumulate gradient from different threads (parallel reduction/hist)
    //
    for(int i=0;i<MAXTHREADS;i++) 
      for(size_t j=0;j<d_mesh.size(); j++)
        d_mesh[j] += grads[i][j];
  }

  void edge_derivatives(
        const TriangleMesh &mesh3d,
        const Img &adjoint,
        const int num_edge_samples,
        DTriangleMesh &d_mesh) 
  {
    // (1) if we have 3d mesh, need to project it to screen for correct edje sampling
    //  
    const TriangleMesh copy   = mesh3d;
    const TriangleMesh* pMesh = &mesh3d;
    
    TriangleMesh localMesh;
    if(mesh3d.m_geomType == GEOM_TYPES::TRIANGLE_3D)
    {
      localMesh = mesh3d;
      for(auto& v : localMesh.vertices) {
        auto vCopy = v;
        VertexShader(*(m_aux.pCamInfo), vCopy.x, vCopy.y, vCopy.z, 
                     v.M);
      }
      localMesh.m_geomType = GEOM_TYPES::TRIANGLE_2D;
      pMesh = &localMesh;
    }
    const TriangleMesh& mesh = *pMesh;
  
    // (2) prepare edjes
    //
    auto edges        = collect_edges(mesh);
    auto edge_sampler = build_edge_sampler(mesh, edges);
  
    // (3) do edje sampling
    // 
    prng::RandomGen gens[MAXTHREADS];
    std::vector<GradReal> grads[MAXTHREADS];
  
    for(int i=0;i<MAXTHREADS;i++)
    {
      gens [i] = prng::RandomGenInit(7777 + i*i + 1);
      grads[i].resize(d_mesh.numVerts()*3);
      memset(grads[i].data(), 0, grads[i].size()*sizeof(GradReal));
    }
  
    //float maxRelativeError = 0.0f;
    #pragma omp parallel for num_threads(MAXTHREADS)
    for (int i = 0; i < num_edge_samples; i++) 
    { 
      auto& gen = gens[omp_get_thread_num()];

      //const float rnd0 = clamp(qmc::rndFloat(i, 0, &g_table[0][0]) + 0.1f*(2.0f*prng::rndFloat(&gen)-1.0f), 0.0f, 1.0f);
      //const float rnd1 = clamp(qmc::rndFloat(i, 1, &g_table[0][0]) + 0.1f*(2.0f*prng::rndFloat(&gen)-1.0f), 0.0f, 1.0f);
      const float rnd0 = prng::rndFloat(&gen);
      const float rnd1 = prng::rndFloat(&gen);
      // pick an edge
      auto edge_id = sample(edge_sampler, rnd0);
      auto edge    = edges[edge_id];
      auto pmf     = edge_sampler.pmf[edge_id];
      
      // pick a point p on the edge
      auto v0 = LiteMath::to_float2(mesh.vertices[edge.v0]);
      auto v1 = LiteMath::to_float2(mesh.vertices[edge.v1]);
      auto t = rnd1;
      auto p = v0 + t * (v1 - v0);
      int xi = int(p.x); 
      int yi = int(p.y); // integer coordinates
      if (xi < 0 || yi < 0 || xi >= adjoint.width() || yi >= adjoint.height()) {
          continue;
      }
      // sample the two sides of the edge
      auto n = normal2D((v1 - v0) / length(v1 - v0));
      
      const float2 coordIn  = p - 1e-3f * n;
      const float2 coordOut = p + 1e-3f * n;
    
      float3 ray_posIn = {0,0,0}, ray_dirIn = {0,0,0};
      float3 ray_posOut = {0,0,0}, ray_dirOut = {0,0,0};
      const auto surfIn    = m_pTracer->CastSingleRay(coordIn.x, coordIn.y, &ray_posIn, &ray_dirIn);
      const auto surfOut   = m_pTracer->CastSingleRay(coordOut.x, coordOut.y, &ray_posOut, &ray_dirOut);

      const auto color_in  = Model::shade(mesh, surfIn, ray_posIn, ray_dirIn);
      const auto color_out = Model::shade(mesh, surfOut, ray_posOut, ray_dirOut);

      // get corresponding adjoint from the adjoint image,
      // multiply with the color difference and divide by the pdf & number of samples.
      float pdf    = pmf  / (length(v1 - v0));
      float weight = 1.0f / (pdf * float(num_edge_samples));
      float adj    = dot(color_in - color_out, adjoint[int2(xi,yi)]);
      // the boundary point is p = v0 + t * (v1 - v0)
      // according to Reynolds transport theorem, the derivatives w.r.t. q is color_diff * dot(n, dp/dq)
      // dp/dv0.x = (1 - t, 0), dp/dv0.y = (0, 1 - t)
      // dp/dv1.x = (    t, 0), dp/dv1.y = (0,     t)
      
      auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight; // v0: (dF/dx_proj, dF/dy_proj)
      auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight; // v1: (dF/dx_proj, dF/dy_proj)
  
      Model::edge_grad(mesh, edge.v0, edge.v1, d_v0, d_v1, m_aux, 
                       grads[omp_get_thread_num()]);
    }    
  
    //std::cout << " (VS_X_grad/VS_Y_grad) maxError = " << maxRelativeError*100.0f << "%" << std::endl;
  
    // accumulate gradient from different threads (parallel reduction/hist)
    //
    for(int i=0;i<MAXTHREADS;i++) 
      for(size_t j=0;j<d_mesh.numVerts()*3; j++)
        d_mesh[j] += grads[i][j];
  }


  std::shared_ptr<IRayTracer> m_pTracer = nullptr;
  int m_samples_per_pixel;
  AuxData m_aux;

  unsigned int m_table[qmc::QRNG_DIMENSIONS][qmc::QRNG_RESOLUTION];
  std::vector<float> m_hammSamples;
};

