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

constexpr static int SAM_PER_PIXEL = 16;
constexpr static int MAXTHREADS    = 8;

unsigned int g_table[qmc::QRNG_DIMENSIONS][qmc::QRNG_RESOLUTION];
float g_hammSamples[2*SAM_PER_PIXEL];

std::shared_ptr<IRayTracer> g_tracer = nullptr;

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////


struct Edge {
    int v0, v1; // vertex ID, v0 < v1

    Edge(int v0, int v1) : v0(min(v0, v1)), v1(max(v0, v1)) {}

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


void render(const TriangleMesh &mesh, int samples_per_pixel,
            Img &img) {

    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    g_tracer->Init(&mesh);

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
            if(mesh.m_meshType == TRIANGLE_VERT_COL)
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
        const vector<Edge> &edges,
        const Sampler &edge_sampler,
        const Img &adjoint,
        const int num_edge_samples,
        Img* screen_dx, Img* screen_dy,
        GradReal* d_vertices) 
{
   
    prng::RandomGen gens[MAXTHREADS];
    for(int i=0;i<MAXTHREADS;i++)
      gens[i] = prng::RandomGenInit(7777 + i*i + 1);

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
      auto v0_3d = mesh.vertices[edge.v0];
      auto v1_3d = mesh.vertices[edge.v1];

      auto v0 = float2(v0_3d.x, v0_3d.y);
      auto v1 = float2(v1_3d.x, v1_3d.y);

      auto t = rnd1;
      auto p = v0 + t * (v1 - v0);
      int xi = int(p.x); 
      int yi = int(p.y); // integer coordinates
      if (xi < 0 || yi < 0 || xi >= adjoint.width() || yi >= adjoint.height()) {
          continue;
      }
      // sample the two sides of the edge
      auto n = normal2D((v1 - v0) / length(v1 - v0));

      //float2 uvIn, uvOut;
      //unsigned faceIdIn = unsigned(-1), faceIdOut = unsigned(-1); 
      //raytrace(mesh, p - 1e-3f * n, &faceIdIn,  &uvIn);
      //raytrace(mesh, p + 1e-3f * n, &faceIdOut, &uvOut);
      
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
      
      auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
      auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;
      
      //auto d_v0 = float3{(1 - t) * n.x, (1 - t) * n.y, (1 - t) * n.z} * adj * weight;
      //auto d_v1 = float3{     t  * n.x,      t  * n.y,      t  * n.z} * adj * weight;

      // if running in parallel, use atomic add here.
      #pragma omp atomic
      d_vertices[edge.v0*3+0] += GradReal(d_v0.x);
      #pragma omp atomic
      d_vertices[edge.v0*3+1] += GradReal(d_v0.y);
      //#pragma omp atomic
      //d_vertices[edge.v0*3+2] += GradReal(d_v0.z);
      
      #pragma omp atomic
      d_vertices[edge.v1*3+0] += GradReal(d_v1.x);
      #pragma omp atomic
      d_vertices[edge.v1*3+1] += GradReal(d_v1.y);
      //#pragma omp atomic
      //d_vertices[edge.v1*3+2] += GradReal(d_v1.z);

      if(screen_dx != nullptr && screen_dy != nullptr) 
      {
        // for the derivatives w.r.t. p, dp/dp.x = (1, 0) and dp/dp.y = (0, 1)
        // the screen space derivatives are the negation of this
        auto dx = -n.x * (color_in - color_out) * weight;
        auto dy = -n.y * (color_in - color_out) * weight;
        // scatter gradients to buffers, in the parallel case, use atomic add here.
        (*screen_dx)[int2(xi, yi)] += dx;
        (*screen_dy)[int2(xi, yi)] += dy;
      }
    }    
}

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              Img* screen_dx,
              Img* screen_dy,
              DTriangleMesh &d_mesh) {

  // (0) Build Acceleration structurres and e.t.c. if needed
  //
  g_tracer->Init(&mesh);

  // (1)
  //
  compute_interior_derivatives(mesh, interior_samples_per_pixel, adjoint, 
                               d_mesh.colors_s());
    
  // (2)
  //
  auto edges        = collect_edges(mesh);
  auto edge_sampler = build_edge_sampler(mesh, edges);
  compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, edge_samples_in_total,
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

  TriangleMesh initialMesh, targetMesh;
  //scn01_TwoTrisFlat(initialMesh, targetMesh);
  scn02_TwoTrisSmooth(initialMesh, targetMesh);

  g_tracer = MakeRayTracer2D("");
  
  if(0) // check gradients with finite difference method
  {
    Img target(img.width(), img.height(), float3{0, 0, 0});
    Img adjoint(img.width(), img.height(), float3{0, 0, 0});
    render(initialMesh, SAM_PER_PIXEL, img);
    render(targetMesh, SAM_PER_PIXEL, target);
    
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad3(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    LossAndDiffLoss(img, target, adjoint); // put MSE ==> adjoint 
    d_render(initialMesh, adjoint, SAM_PER_PIXEL, img.width()*img.height(), nullptr, nullptr, grad1);

    d_finDiff2(initialMesh, "fin_diff", img, target, grad2, 1.0f, 0.01f);
    d_finDiff (initialMesh, "fin_diff", img, target, grad3, 1.0f, 0.01f);
    
    double totalError = 0.0;
    for(size_t i=0;i<grad1.totalParams();i++) {
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1.getData()[i] << "\t";  
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2.getData()[i] << "\t";
      std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad3.getData()[i] << std::endl; 
      double diff = double(grad1.getData()[i] - grad2.getData()[i]);
      totalError += std::abs(diff);
    }
    
    std::cout << "====================================" << std::endl;
    std::cout << "GradError(Total) = " << totalError/double(grad1.totalParams()) << std::endl;
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
