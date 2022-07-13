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

constexpr static int  SAM_PER_PIXEL = 16;
constexpr static int  MAXTHREADS    = 8;
constexpr static bool G_USE3DRT     = false;

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

    //Edge(int v0, int v1) : v0(min(v0, v1)), v1(max(v0, v1)) {}
    Edge(int v0, int v1) : v0(v0), v1(v1) {}

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

  if(mesh.m_meshType == TRIANGLE_VERT_COL)
  {
    const auto  A = mesh.indices[surfInfo.faceId*3+0];
    const auto  B = mesh.indices[surfInfo.faceId*3+1];
    const auto  C = mesh.indices[surfInfo.faceId*3+2];
    const float u = surfInfo.u;
    const float v = surfInfo.v;
    return mesh.colors[A]*u + mesh.colors[B]*v + (1.0f-u-v)*mesh.colors[C]; 
  }
  else
    return mesh.colors[surfInfo.faceId]; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float VS_X(float V[3], const CamInfo& data)
{
  const float W    =  V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = (V[0] * data.projM[0] + V[1] * data.projM[4] + V[2] * data.projM[ 8] + data.projM[12])/W;
  return (xNDC*0.5f + 0.5f)*data.width;
}

static inline float VS_Y(float V[3], const CamInfo& data)
{
  const float W    =   V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = -(V[0] * data.projM[1] + V[1] * data.projM[5] + V[2] * data.projM[ 9] + data.projM[13])/W;
  return (xNDC*0.5f + 0.5f)*data.height;
}

//void VS_X_grad_finDiff(float V[3], const CamInfo &data, float _d_V[3]) 
//{
//  const float epsilon = 5e-5f;
//  float v0 = VS_X(V,data);
//  
//  float V1[3] = {V[0]+epsilon, V[1], V[2]};
//  float V2[3] = {V[0], V[1]+epsilon, V[2]};
//  float V3[3] = {V[0], V[1], V[2]+epsilon};
//
//  float vx = VS_X(V1,data);
//  float vy = VS_X(V2,data);
//  float vz = VS_X(V3,data);
//  
//  _d_V[0] = (vx - v0)/epsilon;
//  _d_V[1] = (vy - v0)/epsilon;
//  _d_V[2] = (vz - v0)/epsilon;
//}
//
//void VS_Y_grad_finDiff(float V[3], const CamInfo &data, float _d_V[3]) 
//{
//  const float epsilon = 5e-5f;
//  float v0 = VS_Y(V,data);
//  
//  float V1[3] = {V[0]+epsilon, V[1], V[2]};
//  float V2[3] = {V[0], V[1]+epsilon, V[2]};
//  float V3[3] = {V[0], V[1], V[2]+epsilon};
//
//  float vx = VS_Y(V1,data);
//  float vy = VS_Y(V2,data);
//  float vz = VS_Y(V3,data);
//  
//  _d_V[0] = (vx - v0)/epsilon;
//  _d_V[1] = (vy - v0)/epsilon;
//  _d_V[2] = (vz - v0)/epsilon;
//}

static inline void VS_X_grad(float V[3], const CamInfo &data, float _d_V[3]) {
    float _t0;
    float _t1;
    float _t2;
    float _t3;
    float _t4;
    float _t5;
    float _d_W = 0;
    float _t6;
    float _t7;
    float _t8;
    float _t9;
    float _t10;
    float _t11;
    float _t12;
    float _t13;
    float _d_xNDC = 0;
    float _t14;
    float _t15;
    _t1 = V[0];
    _t0 = data.projM[3];
    _t3 = V[1];
    _t2 = data.projM[7];
    _t5 = V[2];
    _t4 = data.projM[11];
    const float W = _t1 * _t0 + _t3 * _t2 + _t5 * _t4 + data.projM[15];
    _t8 = V[0];
    _t7 = data.projM[0];
    _t10 = V[1];
    _t9 = data.projM[4];
    _t12 = V[2];
    _t11 = data.projM[8];
    _t13 = (_t8 * _t7 + _t10 * _t9 + _t12 * _t11 + data.projM[12]);
    _t6 = W;
    const float xNDC = _t13 / _t6;
    _t15 = (xNDC * 0.5F + 0.5F);
    _t14 = data.width;
    float VS_X_return = _t15 * _t14;
    {
        float _r14 = 1 * _t14;
        float _r15 = _r14 * 0.5F;
        _d_xNDC += _r15;
        float _r16 = _t15 * 1;
    }
    {
        float _r6 = _d_xNDC / _t6;
        float _r7 = _r6 * _t7;
        _d_V[0] += _r7;
        float _r8 = _t8 * _r6;
        float _r9 = _r6 * _t9;
        _d_V[1] += _r9;
        float _r10 = _t10 * _r6;
        float _r11 = _r6 * _t11;
        _d_V[2] += _r11;
        float _r12 = _t12 * _r6;
        float _r13 = _d_xNDC * -_t13 / (_t6 * _t6);
        _d_W += _r13;
    }
    {
        float _r0 = _d_W * _t0;
        _d_V[0] += _r0;
        float _r1 = _t1 * _d_W;
        float _r2 = _d_W * _t2;
        _d_V[1] += _r2;
        float _r3 = _t3 * _d_W;
        float _r4 = _d_W * _t4;
        _d_V[2] += _r4;
        float _r5 = _t5 * _d_W;
    }
}

static inline void VS_Y_grad(float V[3], const CamInfo &data, float _d_V[3]) {
    float _t0;
    float _t1;
    float _t2;
    float _t3;
    float _t4;
    float _t5;
    float _d_W = 0;
    float _t6;
    float _t7;
    float _t8;
    float _t9;
    float _t10;
    float _t11;
    float _t12;
    float _t13;
    float _d_xNDC = 0;
    float _t14;
    float _t15;
    _t1 = V[0];
    _t0 = data.projM[3];
    _t3 = V[1];
    _t2 = data.projM[7];
    _t5 = V[2];
    _t4 = data.projM[11];
    const float W = _t1 * _t0 + _t3 * _t2 + _t5 * _t4 + data.projM[15];
    _t8 = V[0];
    _t7 = data.projM[1];
    _t10 = V[1];
    _t9 = data.projM[5];
    _t12 = V[2];
    _t11 = data.projM[9];
    _t13 = -(_t8 * _t7 + _t10 * _t9 + _t12 * _t11 + data.projM[13]);
    _t6 = W;
    const float xNDC = _t13 / _t6;
    _t15 = (xNDC * 0.5F + 0.5F);
    _t14 = data.height;
    float VS_Y_return = _t15 * _t14;
    {
        float _r14 = 1 * _t14;
        float _r15 = _r14 * 0.5F;
        _d_xNDC += _r15;
        float _r16 = _t15 * 1;
    }
    {
        float _r6 = _d_xNDC / _t6;
        float _r7 = -_r6 * _t7;
        _d_V[0] += _r7;
        float _r8 = _t8 * -_r6;
        float _r9 = -_r6 * _t9;
        _d_V[1] += _r9;
        float _r10 = _t10 * -_r6;
        float _r11 = -_r6 * _t11;
        _d_V[2] += _r11;
        float _r12 = _t12 * -_r6;
        float _r13 = _d_xNDC * -_t13 / (_t6 * _t6);
        _d_W += _r13;
    }
    {
        float _r0 = _d_W * _t0;
        _d_V[0] += _r0;
        float _r1 = _t1 * _d_W;
        float _r2 = _d_W * _t2;
        _d_V[1] += _r2;
        float _r3 = _t3 * _d_W;
        float _r4 = _d_W * _t4;
        _d_V[2] += _r4;
        float _r5 = _t5 * _d_W;
    }
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

    //#pragma omp parallel for collapse (2)
    for (int y = 0; y < img.height(); y++) { // for each pixel 
      for (int x = 0; x < img.width(); x++) {

        for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
          for (int dx = 0; dx < sqrt_num_samples; dx++) {

            auto xoff = (dx + 0.5f) / float(sqrt_num_samples);
            auto yoff = (dy + 0.5f) / float(sqrt_num_samples);
            auto screen_pos = float2{x + xoff, y + yoff};
            
            auto surf  = g_tracer->CastSingleRay(screen_pos.x, screen_pos.y);
            auto color = shade(mesh, surf);

            img[int2(x,y)] += (color / samples_per_pixel);
          }
        }

      }
    }
}

void compute_interior_derivatives(const TriangleMesh &mesh,
                                  int samples_per_pixel,
                                  const Img &adjoint,
                                  GradReal* d_colors) {
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    //#pragma omp parallel for num_threads(MAXTHREADS)
    for (int y = 0; y < adjoint.height(); y++) { // for each pixel
      for (int x = 0; x < adjoint.width(); x++) {
      
        for (int samId = 0; samId < samples_per_pixel; samId++) // for each subpixel
        {         
          float xoff = g_hammSamples[2*samId+0];
          float yoff = g_hammSamples[2*samId+1];

          //auto screen_pos = float2{x + xoff, y + yoff};
          //float2 uv; unsigned faceIndex = unsigned(-1);
          //raytrace(mesh, screen_pos, &faceIndex, &uv);

          auto surfElem = g_tracer->CastSingleRay(x + xoff, y + yoff);
                    
          if (surfElem.faceId != unsigned(-1)) 
          {          
            auto val = adjoint[int2(x,y)] / samples_per_pixel;
            if(mesh.m_meshType == TRIANGLE_VERT_COL)                // shade_back( => val)
            {
              auto A = mesh.indices[surfElem.faceId*3+0];
              auto B = mesh.indices[surfElem.faceId*3+1];
              auto C = mesh.indices[surfElem.faceId*3+2];
              
              auto contribA = surfElem.u*val;
              auto contribB = surfElem.v*val;
              auto contribC = (1.0f-surfElem.u-surfElem.v)*val;
              #pragma omp atomic
              d_colors[A*3+0] += GradReal(contribA.x);
              #pragma omp atomic
              d_colors[A*3+1] += GradReal(contribA.y);
              #pragma omp atomic
              d_colors[A*3+2] += GradReal(contribA.z);
              
              #pragma omp atomic
              d_colors[B*3+0] += GradReal(contribB.x);
              #pragma omp atomic
              d_colors[B*3+1] += GradReal(contribB.y);
              #pragma omp atomic
              d_colors[B*3+2] += GradReal(contribB.z);
              
              #pragma omp atomic
              d_colors[C*3+0] += GradReal(contribC.x);
              #pragma omp atomic
              d_colors[C*3+1] += GradReal(contribC.y);
              #pragma omp atomic
              d_colors[C*3+2] += GradReal(contribC.z);
            }
            else
            {
              #pragma omp atomic
              d_colors[surfElem.faceId*3+0] += GradReal(val.x); 
              #pragma omp atomic
              d_colors[surfElem.faceId*3+1] += GradReal(val.y);
              #pragma omp atomic
              d_colors[surfElem.faceId*3+2] += GradReal(val.z);
            }
          } //if (faceIndex != unsigned(-1))         
               
        } // for (int samId = 0; samId < samples_per_pixel; samId++)
      }
    }
}

void compute_edge_derivatives(
        const TriangleMesh &mesh,
        const TriangleMesh &mesh3d,
        const vector<Edge> &edges,
        const Sampler &edge_sampler,
        const Img &adjoint,
        const int num_edge_samples, bool a_3dProj,
        Img* screen_dx, Img* screen_dy,
        GradReal* d_vertices) 
{
   
    prng::RandomGen gens[MAXTHREADS];
    for(int i=0;i<MAXTHREADS;i++)
      gens[i] = prng::RandomGenInit(7777 + i*i + 1);

    //float maxRelativeError = 0.0f;

    //#pragma omp parallel for num_threads(MAXTHREADS)
    for (int i = 0; i < num_edge_samples; i++) 
    { 
      auto& gen = gens[omp_get_thread_num()];
      const float rnd0 = clamp(qmc::rndFloat(i, 0, &g_table[0][0]) + 0.1f*(2.0f*prng::rndFloat(&gen)-1.0f), 0.0f, 1.0f);
      const float rnd1 = clamp(qmc::rndFloat(i, 1, &g_table[0][0]) + 0.1f*(2.0f*prng::rndFloat(&gen)-1.0f), 0.0f, 1.0f);

      // pick an edge
      auto edge_id = sample(edge_sampler, rnd0);
      auto edge = edges[edge_id];
      auto pmf = edge_sampler.pmf[edge_id];
      // pick a point p on the edge
      auto v03 = mesh.vertices[edge.v0];
      auto v13 = mesh.vertices[edge.v1];

      auto v0 = float2(v03.x, v03.y);
      auto v1 = float2(v13.x, v13.y);

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
        //if(G_USE3DRT)
        //{
        //  const SurfaceInfo* pSurf = &surfIn;
        //  if(pSurf->faceId == unsigned(-1))
        //    pSurf =  &surfOut;
        //
        //  float minDiff = 10.0f;
        //  const float diffU = std::abs(pSurf->u - t);
        //  const float diffV = std::abs(pSurf->v - t);
        //  const float diffW = std::abs(1.0f - pSurf->u - pSurf->v - t);
        //
        //  if(diffU < minDiff)
        //  {
        //    minDiff = diffU;
        //    t = pSurf->u;
        //  }
        //  if(diffV < minDiff)
        //  {
        //    minDiff = diffV;
        //    t       = pSurf->v;
        //  }
        //  if(diffW < minDiff)
        //  {
        //    minDiff = diffW;
        //    t       = 1.0f - pSurf->u - pSurf->v;
        //  }  
        //}

        auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
        auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;

        float3 v0_dx(0,0,0), v0_dy(0,0,0);
        float3 v1_dx(0,0,0), v1_dy(0,0,0);
        
        float3 v0_3d = mesh3d.vertices[edge.v0];
        float3 v1_3d = mesh3d.vertices[edge.v1];

        VS_X_grad(v0_3d.M, g_uniforms, v0_dx.M);
        VS_Y_grad(v0_3d.M, g_uniforms, v0_dy.M);
        
        VS_X_grad(v1_3d.M, g_uniforms, v1_dx.M);
        VS_Y_grad(v1_3d.M, g_uniforms, v1_dy.M);
        
        //// fin diff check
        //{
        //  float3 v0_dx_check(0,0,0), v0_dy_check(0,0,0);
        //  float3 v1_dx_check(0,0,0), v1_dy_check(0,0,0);
        //  
        //  VS_X_grad_finDiff(v0_3d.M, g_uniforms, v0_dx_check.M);
        //  VS_Y_grad_finDiff(v0_3d.M, g_uniforms, v0_dy_check.M);
        //  
        //  VS_X_grad_finDiff(v1_3d.M, g_uniforms, v1_dx_check.M);
        //  VS_Y_grad_finDiff(v1_3d.M, g_uniforms, v1_dy_check.M);
        //
        //  const float err1 = length(v0_dx - v0_dx_check) / length(v0_dx);
        //  const float err2 = length(v0_dy - v0_dy_check) / length(v0_dy);
        //  const float err3 = length(v1_dx - v1_dx_check) / length(v1_dx);
        //  const float err4 = length(v1_dy - v1_dy_check) / length(v1_dy);
        //  const float errMax = std::max(std::max(err1, err2), std::max(err3, err4));
        //  maxRelativeError = std::max(maxRelativeError, errMax);
        //}

        const float dv0_dx = v0_dx.x*d_v0.x; //  + v0_dx.y*d_v0.y;
        const float dv0_dy = v0_dy.y*d_v0.y; //  + v0_dy.x*d_v0.x;
        const float dv0_dz = (v0_dx.z*d_v0.x + v0_dy.z*d_v0.y); 

        const float dv1_dx = v1_dx.x*d_v1.x; // + v1_dx.y*d_v1.y;
        const float dv1_dy = v1_dy.y*d_v1.y; // + v1_dy.x*d_v1.x;
        const float dv1_dz = (v1_dx.z*d_v1.x + v1_dy.z*d_v1.y); 
  
        // if running in parallel, use atomic add here.
        #pragma omp atomic
        d_vertices[edge.v0*3+0] += GradReal(dv0_dx);
        #pragma omp atomic
        d_vertices[edge.v0*3+1] += GradReal(dv0_dy);
        #pragma omp atomic
        d_vertices[edge.v0*3+2] += GradReal(dv0_dz);
        
        #pragma omp atomic
        d_vertices[edge.v1*3+0] += GradReal(dv1_dx);
        #pragma omp atomic
        d_vertices[edge.v1*3+1] += GradReal(dv1_dy);
        #pragma omp atomic
        d_vertices[edge.v1*3+2] += GradReal(dv1_dz);
      }
      else
      {
        auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
        auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;
  
        // if running in parallel, use atomic add here.
        #pragma omp atomic
        d_vertices[edge.v0*3+0] += GradReal(d_v0.x);
        #pragma omp atomic
        d_vertices[edge.v0*3+1] += GradReal(d_v0.y);
        
        #pragma omp atomic
        d_vertices[edge.v1*3+0] += GradReal(d_v1.x);
        #pragma omp atomic
        d_vertices[edge.v1*3+1] += GradReal(d_v1.y);
      }

      //if(screen_dx != nullptr && screen_dy != nullptr) 
      //{
      //  // for the derivatives w.r.t. p, dp/dp.x = (1, 0) and dp/dp.y = (0, 1)
      //  // the screen space derivatives are the negation of this
      //  auto dx = -n.x * (color_in - color_out) * weight;
      //  auto dy = -n.y * (color_in - color_out) * weight;
      //  // scatter gradients to buffers, in the parallel case, use atomic add here.
      //  (*screen_dx)[int2(xi, yi)] += dx;
      //  (*screen_dy)[int2(xi, yi)] += dy;
      //}
    }    

    //std::cout << " (VS_X_grad/VS_Y_grad) maxError = " << maxRelativeError*100.0f << "%" << std::endl;
}

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              Img* screen_dx,
              Img* screen_dy,
              DTriangleMesh &d_mesh) {

  const TriangleMesh copy   = mesh;
  const TriangleMesh* pMesh = &mesh;
    
  TriangleMesh localMesh;
  if(mesh.m_geomType == TRIANGLE_3D)
  {
    localMesh = mesh;
    for(auto& v : localMesh.vertices) {
      auto vCopy = v;
      v.x = VS_X(vCopy.M, g_uniforms);
      v.y = VS_Y(vCopy.M, g_uniforms);
    }
    localMesh.m_geomType = TRIANGLE_2D;
    pMesh = &localMesh;
  }
  
  // (0) Build Acceleration structurres and e.t.c. if needed
  //
  //if(G_USE3DRT)
  //  g_tracer->Init(&copy);
  //else
  g_tracer->Init(&mesh);
  g_tracer->SetCamera(g_uniforms);

  // (1)
  //
  compute_interior_derivatives(*pMesh, interior_samples_per_pixel, adjoint, 
                               d_mesh.colors_s());
    
  // (2)
  //
  auto edges        = collect_edges(*pMesh);
  auto edge_sampler = build_edge_sampler(*pMesh, edges);
  compute_edge_derivatives(*pMesh, copy, edges, edge_sampler, adjoint, edge_samples_in_total, (d_mesh.m_geomType == TRIANGLE_3D),
                           screen_dx, screen_dy, d_mesh.vertices_s());
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
  //scn03_Triangle3D(initialMesh, targetMesh);
  //scn04_Pyramid3D(initialMesh, targetMesh);
  
  //if(G_USE3DRT)
  //g_tracer = MakeRayTracer3D("");
  g_tracer = MakeRayTracer2D("");
  
  if(0)
  {
    Img initial(img.width(), img.height(), float3{0, 0, 0});
    Img target(img.width(), img.height(), float3{0, 0, 0});
    render(initialMesh, SAM_PER_PIXEL, initial);
    render(targetMesh, SAM_PER_PIXEL, target);
    LiteImage::SaveImage("rendered/initial.bmp", initial);
    LiteImage::SaveImage("rendered/target.bmp", target);
    return 0;
  }

  if(1) // check gradients with finite difference method
  {
    Img target(img.width(), img.height(), float3{0, 0, 0});
    Img adjoint(img.width(), img.height(), float3{0, 0, 0});
    render(initialMesh, SAM_PER_PIXEL, img);
    render(targetMesh, SAM_PER_PIXEL, target);
    
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    LossAndDiffLoss(img, target, adjoint); // put MSE ==> adjoint 
    d_render(initialMesh, adjoint, SAM_PER_PIXEL, img.width()*img.height(), nullptr, nullptr, grad1);
    
    const float dPos = (initialMesh.m_geomType == TRIANGLE_2D) ? 1.0f : 4.0f/float(img.width());

    d_finDiff (initialMesh, "fin_diff", img, target, grad2, dPos, 0.01f);
    
    double totalError = 0.0;
    double posError = 0.0;
    double colError = 0.0;
    bool colorNow = false;
    for(size_t i=0;i<grad1.totalParams();i++) {
      bool colorWasSwitched = colorNow;
      double diff = std::abs(double(grad1.getData()[i] - grad2.getData()[i]));
      if(i < grad1.numVerts()*3)
        posError += diff;
      else
      {
        colorNow = true;
        colError += diff;
      }
      totalError += diff;

      if(!colorWasSwitched && colorNow)
        std::cout << "--------------------------" << std::endl;
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1.getData()[i] << "\t";  
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2.getData()[i] << std::endl;
    }
  
    std::cout << "==========================" << std::endl;
    std::cout << "GradError(vpos ) = " << posError/double(grad1.numVerts()*3) << std::endl;
    std::cout << "GradError(color) = " << colError/double(grad1.numVerts()*3) << std::endl;
    std::cout << "GradError(total) = " << totalError/double(grad1.totalParams()) << std::endl;
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

  pOpt->Init(initialMesh, img); // set different target image

  TriangleMesh mesh3 = pOpt->Run(300);
  img.clear(float3{0,0,0});
  render(mesh3, SAM_PER_PIXEL, img);
  LiteImage::SaveImage("rendered_opt/z_target2.bmp", img);
  
  delete pOpt; pOpt = nullptr;
  return 0;
}
