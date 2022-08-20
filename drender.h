#pragma once

#include "LiteMath.h"
#include "dmesh.h"
#include "dmodels.h"
#include "raytrace.h"

#if DEBUG_RENDER
constexpr static int  MAXTHREADS    = 1;
#else
constexpr static int  MAXTHREADS    = 14;
#endif

#include <omp.h>
#include "qmc.h"
#include <vector>
#include <set>
#include <memory>
#include <iostream>

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

struct IDiffRender
{
  virtual void commit(const TriangleMesh &mesh) = 0;
  virtual void render(const TriangleMesh &mesh, const CamInfo* cams, Img *imgames, int a_viewsNum) = 0;
  virtual void d_render(const TriangleMesh &mesh, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                        DTriangleMesh &d_mesh,
                        Img* debugImages, int debugImageNum) = 0;
};

template<class Model>
struct DiffRender : public IDiffRender
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
  
  
  void commit(const TriangleMesh &mesh) 
  {
    m_pTracer->Init(&mesh); // Build Acceleration structurres and e.t.c. if needed
    m_pLastPreparedMesh = &mesh;
  }

  void render(const TriangleMesh &mesh, const CamInfo* cams, Img *imgames, int a_viewsNum) override // TODO: add BSPImage rendering
  {
    auto sqrt_num_samples  = (int)sqrt((float)m_samples_per_pixel);
    auto samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    if(&mesh != m_pLastPreparedMesh)
    {
      std::cout << "[DiffRender::render]: error, renderer was not prepared for this mesh!" << std::endl;
      return;
    }
    
    for(int camId =0; camId<a_viewsNum; camId++) { // TODO: also can make parallel if m_pTracer->clone() is implemented

      const CamInfo& cam = cams[camId];
      Img&           img = imgames[camId];

      m_pTracer->SetCamera(cam);
      
      #if (DEBUG_RENDER==0)
      #pragma omp parallel for collapse (2) num_threads(MAXTHREADS) 
      #endif 
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
  }

  
  void d_render(const TriangleMesh &mesh, const CamInfo* cams, const Img *adjoints, int a_viewsNum, const int edge_samples_in_total,
                DTriangleMesh &d_mesh,
                Img* debugImages, int debugImageNum) override
  {  
    if(&mesh != m_pLastPreparedMesh)
    {
      std::cout << "[DiffRender::render]: error, renderer was not prepared for this mesh!" << std::endl;
      return;
    }

    for(int camId=0; camId<a_viewsNum; camId++) {
      
      m_pTracer->SetCamera(cams[camId]);
  
      m_aux.pCamInfo      = cams + camId;
      m_aux.debugImages   = debugImages;
      m_aux.debugImageNum = debugImageNum;
    
      interior_derivatives(mesh, adjoints[camId], d_mesh);
  
      edge_derivatives(mesh, adjoints[camId], edge_samples_in_total, d_mesh);
    }
  }

private:

  const TriangleMesh* m_pLastPreparedMesh = nullptr;

  void interior_derivatives(const TriangleMesh &mesh, const Img &adjoint,
                            DTriangleMesh &d_mesh) 
  {
    auto sqrt_num_samples = (int)sqrt((float)m_samples_per_pixel);
    auto samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
  
    DTriangleMesh grads[MAXTHREADS];
    for(int i=0;i<MAXTHREADS;i++) {
      grads[i] = d_mesh;
      grads[i].clear(); // TODO: make this more effitiient
    }
    
    #if (DEBUG_RENDER==0)
    #pragma omp parallel for collapse (2) num_threads(MAXTHREADS)
    #endif
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
    #if (DEBUG_RENDER==0)
    #pragma omp parallel for num_threads(MAXTHREADS)
    #endif
    for (int i = 0; i < num_edge_samples; i++) 
    { 
      auto& gen = gens[omp_get_thread_num()];

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
    
      float3 ray_posIn   = {0,0,0}, ray_dirIn  = {0,0,0};
      float3 ray_posOut  = {0,0,0}, ray_dirOut = {0,0,0};
      const auto surfIn  = m_pTracer->CastSingleRay(coordIn.x, coordIn.y, &ray_posIn, &ray_dirIn);
      const auto surfOut = m_pTracer->CastSingleRay(coordOut.x, coordOut.y, &ray_posOut, &ray_dirOut);

      const auto color_in  = Model::shade(mesh, surfIn, ray_posIn, ray_dirIn);
      const auto color_out = Model::shade(mesh, surfOut, ray_posOut, ray_dirOut);

      // get corresponding adjoint from the adjoint image,
      // multiply with the color difference and divide by the pdf & number of samples.
      float pdf    = pmf  / (length(v1 - v0));
      float weight = 1.0f / (pdf * float(num_edge_samples));
      float adj    = dot(color_in - color_out, adjoint[int2(xi,yi)]);
      
      if(adj*adj > 0.0f)
      {
        // the boundary point is p = v0 + t * (v1 - v0)
        // according to Reynolds transport theorem, the derivatives w.r.t. q is color_diff * dot(n, dp/dq)
        // dp/dv0.x = (1 - t, 0), dp/dv0.y = (0, 1 - t)
        // dp/dv1.x = (    t, 0), dp/dv1.y = (0,     t)
        
        auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight; // v0: (dF/dx_proj, dF/dy_proj)
        auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight; // v1: (dF/dx_proj, dF/dy_proj)
        
        Model::edge_grad(mesh3d, edge.v0, edge.v1, d_v0, d_v1, m_aux, 
                         grads[omp_get_thread_num()]);
      }
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


std::shared_ptr<IDiffRender> MakeDifferentialRenderer(const TriangleMesh &a_mesh, int a_samplesPerPixel);
