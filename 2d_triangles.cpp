#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include "LiteMath.h"

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

#include <cassert>
#include <iomanip>

#include "dmesh.h"
#include "functions.h"
#include "raytrace.h"

#include "optimizer.h"
#include "scenes.h"

#include "qmc.h"

using std::for_each;
using std::upper_bound;
using std::vector;
using std::string;
using std::min;
using std::max;
using std::set;
using std::fstream;

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::int2;

using LiteMath::clamp;
using LiteMath::normalize;

#define DEBUG_RENDER 1

#if DEBUG_RENDER
constexpr static int  MAXTHREADS    = 1;
#else
constexpr static int  MAXTHREADS    = 14;
#endif

constexpr static int  SAM_PER_PIXEL = 16;
constexpr static int  G_DEBUG_VERT_ID = 0;

unsigned int g_table[qmc::QRNG_DIMENSIONS][qmc::QRNG_RESOLUTION];
float g_hammSamples[2*SAM_PER_PIXEL];

std::shared_ptr<IRayTracer> g_tracer = nullptr;
CamInfo g_uniforms;

void glhFrustumf3(float *matrix, float left, float right, float bottom, float top, float znear, float zfar)
{
  float temp, temp2, temp3, temp4;
  temp = 2.0f * znear;
  temp2 = right - left;
  temp3 = top - bottom;
  temp4 = zfar - znear;
  matrix[0] = temp / temp2;
  matrix[1] = 0.0;
  matrix[2] = 0.0;
  matrix[3] = 0.0;
  matrix[4] = 0.0;
  matrix[5] = temp / temp3;
  matrix[6] = 0.0;
  matrix[7] = 0.0;
  matrix[8] = (right + left) / temp2;
  matrix[9] = (top + bottom) / temp3;
  matrix[10] = (-zfar - znear) / temp4;
  matrix[11] = -1.0;
  matrix[12] = 0.0;
  matrix[13] = 0.0;
  matrix[14] = (-temp * zfar) / temp4;
  matrix[15] = 0.0;
}

// matrix will receive the calculated perspective matrix. You would have to upload to your shader or use glLoadMatrixf if you aren't using shaders
//
void glhPerspectivef3(float *matrix, float fovy, float aspectRatio, float znear, float zfar)
{
  const float ymax = znear * std::tan(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspectRatio;
  glhFrustumf3(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
}

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


struct Edge {
    int v0, v1; // vertex ID, v0 < v1

    Edge(int a_v0, int a_v1) : v0(min(a_v0, a_v1)), v1(max(a_v0, a_v1)) {}
    //Edge(int a_v0, int a_v1) : v0(a_v0), v1(a_v1) {}

    // for sorting edges
    bool operator<(const Edge &e) const {
        return this->v0 != e.v0 ? this->v0 < e.v0 : this->v1 < e.v1;
    }
};

// for sampling edges with inverse transform sampling
struct Sampler {
    vector<float> pmf; // probability mass function
    vector<float> cdf;
};

// build a discrete CDF using edge length
Sampler build_edge_sampler(const TriangleMesh &mesh,
                           const vector<Edge> &edges) {
    vector<float> pmf;
    vector<float> cdf;
    pmf.reserve(edges.size());
    cdf.reserve(edges.size() + 1);
    cdf.push_back(0);
    for (auto edge : edges) {
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        pmf.push_back(length(v1 - v0));
        cdf.push_back(pmf.back() + cdf.back());
    }
    auto length_sum = cdf.back();
    for_each(pmf.begin(), pmf.end(), [&](float &p) {p /= length_sum;});
    for_each(cdf.begin(), cdf.end(), [&](float &p) {p /= length_sum;});
    return Sampler{pmf, cdf};
}

// binary search for inverting the CDF in the sampler
int sample(const Sampler &sampler, const float u) {
    auto cdf = sampler.cdf;
    return clamp(upper_bound(
        cdf.begin(), cdf.end(), u) - cdf.begin() - 1,
        0, cdf.size() - 2);
}

// given a triangle mesh, collect all edges.
vector<Edge> collect_edges(const TriangleMesh &mesh) {
    set<Edge> edges;
    for (size_t i=0; i<mesh.indices.size();i+=3) 
    {
      auto A = mesh.indices[i+0];
      auto B = mesh.indices[i+1];
      auto C = mesh.indices[i+2];  
      edges.insert(Edge(A, B));
      edges.insert(Edge(B, C));
      edges.insert(Edge(C, A));
    }
    return vector<Edge>(edges.begin(), edges.end());
}


inline float3 shade(const TriangleMesh &mesh, const SurfaceInfo& surfInfo)
{
  if (surfInfo.faceId == unsigned(-1))
    return float3(0,0,0); // BGCOLOR

  if(mesh.m_meshType == MESH_TYPES::TRIANGLE_VERT_COL)
  {
    const auto  A = mesh.indices[surfInfo.faceId*3+0];
    const auto  B = mesh.indices[surfInfo.faceId*3+1];
    const auto  C = mesh.indices[surfInfo.faceId*3+2];
    const float u = surfInfo.u;
    const float v = surfInfo.v;
    return mesh.colors[A]*(1.0f-u-v) + mesh.colors[B]*v + u*mesh.colors[C]; 
  }
  else
    return mesh.colors[surfInfo.faceId]; 
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void render(const TriangleMesh &mesh, int samples_per_pixel,
            Img &img) 
{
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    g_tracer->Init(&mesh);
    g_tracer->SetCamera(g_uniforms);

    #pragma omp parallel for collapse (2) num_threads(MAXTHREADS) 
    for (int y = 0; y < img.height(); y++) { // for each pixel 
      for (int x = 0; x < img.width(); x++) {
        
        float3 pixelColor(0,0,0);

        for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
          for (int dx = 0; dx < sqrt_num_samples; dx++) {

            auto xoff = (dx + 0.5f) / float(sqrt_num_samples);
            auto yoff = (dy + 0.5f) / float(sqrt_num_samples);
            auto screen_pos = float2{x + xoff, y + yoff};
            
            auto surf  = g_tracer->CastSingleRay(screen_pos.x, screen_pos.y);
            auto color = shade(mesh, surf);

            pixelColor += (color / samples_per_pixel);
          }
        }

        img[int2(x,y)] = pixelColor;
      }
    }
}

void compute_interior_derivatives(const TriangleMesh &mesh,
                                  int samples_per_pixel,
                                  const Img &adjoint,
                                  DTriangleMesh &d_mesh,
                                  Img* debugImages, int debugImageNum) 
{
  auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
  samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
  
  DTriangleMesh grads[MAXTHREADS];
  for(int i=0;i<MAXTHREADS;i++) {
    grads[i] = d_mesh;
    //grads[i].clear(); // don't have to do this explicitly because 'compute_interior_derivatives' is called in the first pass and gradient is already 0.0
  }

  #pragma omp parallel for collapse (2) num_threads(MAXTHREADS)
  for (int y = 0; y < adjoint.height(); y++) // for each pixel
  { 
    for (int x = 0; x < adjoint.width(); x++) 
    {
      GradReal* d_colors = grads[omp_get_thread_num()].colors_s();
      GradReal* d_pos    = grads[omp_get_thread_num()].vertices_s();
      for (int samId = 0; samId < samples_per_pixel; samId++) // for each subpixel
      {         
        float xoff = g_hammSamples[2*samId+0];
        float yoff = g_hammSamples[2*samId+1];
        float3 ray_pos = {0,0,0}, ray_dir = {0,0,0};
        auto surfElem = g_tracer->CastSingleRay(x + xoff, y + yoff, &ray_pos, &ray_dir);
        //if(x == 110 && y == 100)
        //{
        //  int a = 2;
        //}
        if (surfElem.faceId != unsigned(-1)) 
        {          
          auto val = adjoint[int2(x,y)] / samples_per_pixel;
          if(mesh.m_meshType == MESH_TYPES::TRIANGLE_VERT_COL)                // shade_back( => val)
          {
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
          
            if(0) // backpropagate color change to positions
            {
              const float3 c0 = mesh.colors[A];
              const float3 c1 = mesh.colors[B];
              const float3 c2 = mesh.colors[C];
              
              //const float dF_dU = dot((c0-c2), val);
              //const float dF_dV = dot((c1-c2), val);

              const float dF_dU = dot((c2-c0)+(c2-c1), val);
              const float dF_dV = dot((c1-c0), val);
              
              if(dF_dU > 0.0f || dF_dV > 0.0f) 
              {
                const float3 v0 = mesh.vertices[A];
                const float3 v1 = mesh.vertices[B];
                const float3 v2 = mesh.vertices[C];
                float3 dU_dvert[3] = {};
                float3 dV_dvert[3] = {};
                
                BarU_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, /* --> */ dU_dvert[0].M, dU_dvert[1].M, dU_dvert[2].M);
                BarV_grad(ray_pos.M, ray_dir.M, v0.M, v1.M, v2.M, /* --> */ dV_dvert[0].M, dV_dvert[1].M, dV_dvert[2].M);
              
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
          }
          else
          {
            d_colors[surfElem.faceId*3+0] += GradReal(val.x); 
            d_colors[surfElem.faceId*3+1] += GradReal(val.y);
            d_colors[surfElem.faceId*3+2] += GradReal(val.z);
          }
        } //if (faceIndex != unsigned(-1))         
             
      } // for (int samId = 0; samId < samples_per_pixel; samId++)
    }
  }
  
  // accumulate gradient from different threads (parallel reduction/hist)
  //
  for(int i=0;i<MAXTHREADS;i++) 
    for(size_t j=0;j<d_mesh.size(); j++)
      d_mesh[j] += grads[i][j];
}

void compute_edge_derivatives(
        const TriangleMesh &mesh,
        const TriangleMesh &mesh3d,
        const vector<Edge> &edges,
        const Sampler &edge_sampler,
        const Img &adjoint,
        const int num_edge_samples, bool a_3dProj,
        DTriangleMesh &d_mesh,
        Img* debugImages, int debugImageNum) 
{
   
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
    //GradReal* d_vertices = grads[omp_get_thread_num()].vertices_s();
    GradReal* d_vertices = grads[omp_get_thread_num()].data();
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
    
    const auto surfIn    = g_tracer->CastSingleRay(coordIn.x, coordIn.y);
    const auto surfOut   = g_tracer->CastSingleRay(coordOut.x, coordOut.y);
    const auto color_in  = shade(mesh, surfIn);
    const auto color_out = shade(mesh, surfOut);
    // get corresponding adjoint from the adjoint image,
    // multiply with the color difference and divide by the pdf & number of samples.
    float pdf    = pmf  / (length(v1 - v0));
    float weight = 1.0f / (pdf * float(num_edge_samples));
    float adj    = dot(color_in - color_out, adjoint[int2(xi,yi)]);
    // the boundary point is p = v0 + t * (v1 - v0)
    // according to Reynolds transport theorem, the derivatives w.r.t. q is color_diff * dot(n, dp/dq)
    // dp/dv0.x = (1 - t, 0), dp/dv0.y = (0, 1 - t)
    // dp/dv1.x = (    t, 0), dp/dv1.y = (0,     t)
    
    if(a_3dProj) 
    {
      auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight; // v0: (dF/dx_proj, dF/dy_proj)
      auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight; // v1: (dF/dx_proj, dF/dy_proj)
      float3 v0_d[2] = {{0,0,0},{0,0,0}}; 
      float3 v1_d[2] = {{0,0,0},{0,0,0}}; 
      
      float3 v0_3d = mesh3d.vertices[edge.v0];
      float3 v1_3d = mesh3d.vertices[edge.v1];
      VS_X_grad(v0_3d.M, g_uniforms, v0_d[0].M);
      VS_Y_grad(v0_3d.M, g_uniforms, v0_d[1].M);
      VS_X_grad(v1_3d.M, g_uniforms, v1_d[0].M);
      VS_Y_grad(v1_3d.M, g_uniforms, v1_d[1].M);
      
      //float temp[2]={};
      //VertexShader_jac(g_uniforms, v0_3d.x, v0_3d.y, v0_3d.z, temp, v0_d[0].M);
      //VertexShader_jac(g_uniforms, v1_3d.x, v1_3d.y, v1_3d.z, temp, v1_d[0].M); 
       
      const float dv0_dx = v0_d[0].x*d_v0.x; //  + v0_dx.y*d_v0.y;
      const float dv0_dy = v0_d[1].y*d_v0.y; //  + v0_dy.x*d_v0.x;
      const float dv0_dz = (v0_d[0].z*d_v0.x + v0_d[1].z*d_v0.y); 
       
      const float dv1_dx = v1_d[0].x*d_v1.x; // + v1_dx.y*d_v1.y;
      const float dv1_dy = v1_d[1].y*d_v1.y; // + v1_dy.x*d_v1.x;
      const float dv1_dz = (v1_d[0].z*d_v1.x + v1_d[1].z*d_v1.y); 
      
      #if DEBUG_RENDER
      for(int debugId=0; debugId<3; debugId++) 
      {
        if(G_DEBUG_VERT_ID + debugId == edge.v0)
        {
          if(debugImageNum > 0 && debugImages!= nullptr)
            debugImages[debugId][int2(xi,yi)] += float3(dv0_dx,dv0_dy,dv0_dz);
        }
        else if(G_DEBUG_VERT_ID + debugId == edge.v1)
        {
          if(debugImageNum > 0)
            debugImages[debugId][int2(xi,yi)] += float3(dv1_dx,dv1_dy,dv1_dz);
        }
      }
      #endif
      // if running in parallel, use atomic add here.
      d_vertices[edge.v0*3+0] += GradReal(dv0_dx);
      d_vertices[edge.v0*3+1] += GradReal(dv0_dy);
      d_vertices[edge.v0*3+2] += GradReal(dv0_dz);
      
      d_vertices[edge.v1*3+0] += GradReal(dv1_dx);
      d_vertices[edge.v1*3+1] += GradReal(dv1_dy);
      d_vertices[edge.v1*3+2] += GradReal(dv1_dz);
    }
    else
    {
      auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
      auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;

      // if running in parallel, use atomic add here.
      d_vertices[edge.v0*3+0] += GradReal(d_v0.x);
      d_vertices[edge.v0*3+1] += GradReal(d_v0.y);
      
      d_vertices[edge.v1*3+0] += GradReal(d_v1.x);
      d_vertices[edge.v1*3+1] += GradReal(d_v1.y);
    }
  }    

  //std::cout << " (VS_X_grad/VS_Y_grad) maxError = " << maxRelativeError*100.0f << "%" << std::endl;

  // accumulate gradient from different threads (parallel reduction/hist)
  //
  for(int i=0;i<MAXTHREADS;i++) 
    for(size_t j=0;j<d_mesh.numVerts()*3; j++)
      d_mesh[j] += grads[i][j];
}

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              DTriangleMesh &d_mesh,
              Img* debugImages, int debugImageNum) {

  const TriangleMesh copy   = mesh;
  const TriangleMesh* pMesh = &mesh;
    
  TriangleMesh localMesh;
  if(mesh.m_geomType == GEOM_TYPES::TRIANGLE_3D)
  {
    localMesh = mesh;
    for(auto& v : localMesh.vertices) {
      auto vCopy = v;
      VertexShader(g_uniforms, vCopy.x, vCopy.y, vCopy.z, 
                   v.M);
    }
    localMesh.m_geomType = GEOM_TYPES::TRIANGLE_2D;
    pMesh = &localMesh;
  }
  
  // (0) Build Acceleration structurres and e.t.c. if needed
  //
  g_tracer->Init(&mesh);
  g_tracer->SetCamera(g_uniforms);

  // (1)
  //
  compute_interior_derivatives(copy, interior_samples_per_pixel, adjoint, // pass always 3d mesh?
                               d_mesh, debugImages, debugImageNum);
    
  // (2)
  //
  auto edges        = collect_edges(*pMesh);
  auto edge_sampler = build_edge_sampler(*pMesh, edges);
  compute_edge_derivatives(*pMesh, copy, edges, edge_sampler, adjoint, edge_samples_in_total, (d_mesh.m_geomType == GEOM_TYPES::TRIANGLE_3D),
                           d_mesh, debugImages, debugImageNum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PrintMesh(const DTriangleMesh& a_mesh)
{
  for(int i=0; i<a_mesh.numVerts();i++)
    std::cout << "ver[" << i << "]: " << a_mesh.vert_at(i).x << ", " << a_mesh.vert_at(i).y << std::endl;  
  std::cout << std::endl;
  for(size_t i=0; i<a_mesh.numFaces();i++)
    std::cout << "col[" << i << "]: " << a_mesh.color_at(i).x << ", " << a_mesh.color_at(i).y << ", " << a_mesh.color_at(i).z << std::endl;
  std::cout << std::endl;
}


float LossAndDiffLoss(const Img& b, const Img& a, Img& a_outDiff)
{
  assert(a.width()*a.height() == b.width()*b.height());
  double accumMSE = 0.0f;
  const size_t imgSize = a.width()*a.height();
  for(size_t i=0;i<imgSize;i++)
  {
    const float3 diffVec = b.data()[i] - a.data()[i];
    a_outDiff.data()[i] = 2.0f*diffVec;                    // (I[x,y] - I_target[x,y])    // dirrerential of the loss function 
    accumMSE += double(dot(diffVec, diffVec));             // (I[x,y] - I_target[x,y])^2  // the loss function itself
  }
  return float(accumMSE);
}

float MSE(const Img& b, const Img& a) { return LiteImage::MSE(b,a); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);


void d_finDiff2(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IOptimizer* CreateSimpleOptimizer(); 
IOptimizer* CreateComplexOptimizer();

int main(int argc, char *argv[]) 
{
  #ifdef WIN32
  mkdir("rendered");
  mkdir("rendered_opt");
  mkdir("fin_diff");
  #else
  mkdir("rendered",     S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir("rendered_opt", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir("fin_diff",     S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  qmc::init(g_table);
  qmc::planeHammersley(g_hammSamples, SAM_PER_PIXEL);

  Img img(256, 256);

  g_uniforms.width  = float(img.width());
  g_uniforms.height = float(img.height());
  glhPerspectivef3(g_uniforms.projM, 45.0f, g_uniforms.width / g_uniforms.height, 0.1f, 100.0f);

  TriangleMesh initialMesh, targetMesh;
  //scn01_TwoTrisFlat(initialMesh, targetMesh);
  scn02_TwoTrisSmooth(initialMesh, targetMesh);
  //scn03_Triangle3D_White(initialMesh, targetMesh);
  //scn04_Triangle3D_Colored(initialMesh, targetMesh);
  //scn05_Pyramid3D(initialMesh, targetMesh);

  g_tracer = MakeRayTracer2D("");  
  //g_tracer = MakeRayTracer3D("");

  if(1)
  {
    Img initial(img.width(), img.height(), float3{0, 0, 0});
    Img target(img.width(), img.height(), float3{0, 0, 0});
    render(initialMesh, SAM_PER_PIXEL, initial);
    render(targetMesh, SAM_PER_PIXEL, target);
    LiteImage::SaveImage("rendered/initial.bmp", initial);
    LiteImage::SaveImage("rendered/target.bmp", target);
    //return 0;
  }

  if(1) // check gradients with finite difference method
  {
    Img target(img.width(), img.height(), float3{0, 0, 0});
    Img adjoint(img.width(), img.height(), float3{0, 0, 0});
    
    Img dxyzDebug[3];
    for(int i=0;i<3;i++)
      dxyzDebug[i] = Img(img.width(), img.height(), float3{0, 0, 0});

    render(initialMesh, SAM_PER_PIXEL, img);
    render(targetMesh, SAM_PER_PIXEL, target);
    
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    LossAndDiffLoss(img, target, adjoint); // put MSE ==> adjoint 
    d_render(initialMesh, adjoint, SAM_PER_PIXEL, img.width()*img.height(), 
             grad1, dxyzDebug, 3);
    
    for(int i=0;i<3;i++)
    {
      std::stringstream strOut;
      strOut << "our_diff/pos_xyz_" << G_DEBUG_VERT_ID+i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), dxyzDebug[i]);
    }

    const float dPos = (initialMesh.m_geomType == GEOM_TYPES::TRIANGLE_2D) ? 1.0f : 2.0f/float(img.width());
    //d_finDiff (initialMesh, "fin_diff", img, target, grad2, dPos, 0.01f);
    d_finDiff2(initialMesh, "fin_diff", img, target, grad2, dPos, 0.01f);
    
    double totalError = 0.0;
    double posError = 0.0;
    double colError = 0.0;
    double posLengthL1 = 0.0;
    double colLengthL1 = 0.0;

    auto subvecPos1 = grad1.subvecPos();
    auto subvecCol1 = grad1.subvecCol();

    auto subvecPos2 = grad2.subvecPos();
    auto subvecCol2 = grad2.subvecCol();

    for(size_t i=0;i<subvecPos1.size();i++) {
      double diff = std::abs(double(subvecPos1[i] - subvecPos2[i]));
      posError    += diff;
      totalError  += diff;
      posLengthL1 += std::abs(subvecPos2[i]);
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[i] << "\t";  
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[i] << std::endl;
    }

    std::cout << "--------------------------" << std::endl;
    for(size_t i=0;i<subvecCol1.size();i++) {
      double diff = std::abs(double(subvecCol1[i] - subvecCol2[i]));
      colError   += diff;
      totalError += diff;
      colLengthL1 += std::abs(subvecCol2[i]);
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[subvecPos1.size() + i] << "\t";  
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[subvecPos1.size() + i] << std::endl;
    }
  
    std::cout << "==========================" << std::endl;
    std::cout << "GradErr[L1](vpos ) = " << posError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(posError/posLengthL1) << "%" << std::endl;
    std::cout << "GradErr[L1](color) = " << colError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(colError/colLengthL1) << "%" << std::endl;
    std::cout << "GradErr[L1](total) = " << totalError/double(grad1.size()) << "\t which is \t" << 100.0*(totalError/(posLengthL1+colLengthL1)) << "%" << std::endl;
    return 0;
  }

  img.clear(float3{0,0,0});
  render(targetMesh, SAM_PER_PIXEL, img);
  LiteImage::SaveImage("rendered_opt/z_target.bmp", img);
  
  #ifdef COMPLEX_OPT
  IOptimizer* pOpt = CreateComplexOptimizer();
  #else
  IOptimizer* pOpt = CreateSimpleOptimizer();
  #endif

  //pOpt->Init(initialMesh, img, {30,GD_Naive}); 
  pOpt->Init(initialMesh, img, {100,GD_Adam}); 

  TriangleMesh mesh3 = pOpt->Run(300);
  img.clear(float3{0,0,0});
  render(mesh3, SAM_PER_PIXEL, img);
  LiteImage::SaveImage("rendered_opt/z_target2.bmp", img);
  
  delete pOpt; pOpt = nullptr;
  return 0;
}
