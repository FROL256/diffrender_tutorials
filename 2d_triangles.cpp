#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "Bitmap.h"
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
#include "optimizer.h"
#include "scenes.h"

using std::for_each;
using std::upper_bound;
using std::vector;
using std::string;
using std::uniform_real_distribution;
using std::min;
using std::max;
using std::set;
using std::fstream;
using std::mt19937;
uniform_real_distribution<float> uni_dist(0, 1);

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::int2;

using LiteMath::clamp;
using LiteMath::normalize;

float2 normal(const float2 &v) {return float2{-v.y, v.x};} // Vec2f normal(const Vec2f &v) {return Vec2f{-v.y, v.x};}

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

static inline unsigned IntColorUint32(int r, int g, int b)
{
  return unsigned(r | (g << 8) | (b << 16) | 0xFF000000);
}

static inline int tonemap(float x) { return int(pow(clamp(x, float(0), float(1)), float(1/2.2))*255 + float(.5)); }

void save_img(const Img &img, const string &filename, bool flip) 
{
  std::vector<unsigned> colors(img.width * img.height);
  for (int y = 0; y < img.height; y++)
  {
    const int offset1 = (img.height - y - 1)*img.width;
    const int offset2 = y*img.width;
    for(int x =0; x < img.width; x++)
    {
      auto c = flip ? (-1.0f)*img.color[offset1 + x] : img.color[offset1 + x];
      colors[offset2+x] = IntColorUint32(tonemap(c.x), tonemap(c.y), tonemap(c.z));
    }
  }
  
  SaveBMP(filename.c_str(), colors.data(), img.width, img.height);
}

static inline float edgeFunction(float2 a, float2 b, float2 c) // actuattly just a mixed product ... :)
{
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

// trace a single ray at screen_pos, intersect with the triangle mesh.
float3 raytrace(const TriangleMesh &mesh, const float2 &screen_pos,
                unsigned *out_faceIndex = nullptr, float2* uv = nullptr) 
{

  // loop over all triangles in a mesh, return the first one that hits
  for (size_t i = 0; i < (int)mesh.indices.size(); i+=3) 
  {
    // retrieve the three vertices of a triangle
    auto A = mesh.indices[i+0];
    auto B = mesh.indices[i+1];
    auto C = mesh.indices[i+2];
    
    auto v0 = mesh.vertices[A];
    auto v1 = mesh.vertices[B];
    auto v2 = mesh.vertices[C];

    // form three half-planes: v1-v0, v2-v1, v0-v2
    // if a point is on the same side of all three half-planes, it's inside the triangle.
    auto n01 = normal(v1 - v0);
    auto n12 = normal(v2 - v1);
    auto n20 = normal(v0 - v2);
    
    const bool side01 = dot(screen_pos - v0, n01) > 0;
    const bool side12 = dot(screen_pos - v1, n12) > 0;
    const bool side20 = dot(screen_pos - v2, n20) > 0;
    if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) 
    {
      if (out_faceIndex != nullptr) 
        *out_faceIndex = i/3; // because we store face id here
      
      const float areaInv = 1.0f / edgeFunction(v0, v1, v2); 
      const float e0      = edgeFunction(v0, v1, screen_pos);
      const float e1      = edgeFunction(v1, v2, screen_pos);
      const float e2      = edgeFunction(v2, v0, screen_pos);
      const float u = e1*areaInv; // v0
      const float v = e2*areaInv; // v1 

      if(uv != nullptr)
        *uv = float2(u,v);

      if(mesh.type == TRIANGLE_2D_VERT_COL)
        return mesh.colors[A]*u + mesh.colors[B]*v + (1.0f-u-v)*mesh.colors[C]; 
      else
        return mesh.colors[i/3]; 
    }
  }

  // return background
  if (out_faceIndex != nullptr)
    *out_faceIndex = unsigned(-1);

  return float3{0, 0, 0};
}

void render(const TriangleMesh &mesh,
            int samples_per_pixel,
            mt19937 &rng,
            Img &img) {
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    for (int y = 0; y < img.height; y++) { // for each pixel
        for (int x = 0; x < img.width; x++) {
            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = float2{x + xoff, y + yoff};
                    float2 uv;
                    auto color = raytrace(mesh, screen_pos, nullptr, &uv);     
                    img.color[y * img.width + x] += color / samples_per_pixel; //#TODO: atomics
                }
            }
        }
    }
}

void compute_interior_derivatives(const TriangleMesh &mesh,
                                  int samples_per_pixel,
                                  const Img &adjoint,
                                  mt19937 &rng,
                                  float3* d_colors) {
    auto sqrt_num_samples = (int)sqrt((float)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    
    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {
            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = float2{x + xoff, y + yoff};
                    unsigned faceIndex = unsigned(-1);
                    float2 uv;
                    raytrace(mesh, screen_pos, &faceIndex, &uv);
                    if (faceIndex != unsigned(-1)) 
                    {
                      auto val = adjoint.color[y * adjoint.width + x] / samples_per_pixel;

                      if(mesh.type == TRIANGLE_2D_VERT_COL)
                      {
                        auto A = mesh.indices[faceIndex*3+0];
                        auto B = mesh.indices[faceIndex*3+1];
                        auto C = mesh.indices[faceIndex*3+2];
                        d_colors[A] += uv.x*val;
                        d_colors[B] += uv.y*val;
                        d_colors[C] += (1.0f-uv.x-uv.y)*val;
                      }
                      else
                        d_colors[faceIndex] += val; // if running in parallel, use atomic add here.
                    }
                }
            }
        }
    }
}

void compute_edge_derivatives(
        const TriangleMesh &mesh,
        const vector<Edge> &edges,
        const Sampler &edge_sampler,
        const Img &adjoint,
        const int num_edge_samples,
        mt19937 &rng,
        Img* screen_dx, Img* screen_dy,
        float2* d_vertices, float3* d_colors = nullptr) {

    
    for (int i = 0; i < num_edge_samples; i++) 
    {
      // pick an edge
      auto edge_id = sample(edge_sampler, uni_dist(rng));
      auto edge = edges[edge_id];
      auto pmf = edge_sampler.pmf[edge_id];
      // pick a point p on the edge
      auto v0 = mesh.vertices[edge.v0];
      auto v1 = mesh.vertices[edge.v1];
      auto t = uni_dist(rng);
      auto p = v0 + t * (v1 - v0);
      int xi = int(p.x); 
      int yi = int(p.y); // integer coordinates
      if (xi < 0 || yi < 0 || xi >= adjoint.width || yi >= adjoint.height) {
          continue;
      }
      // sample the two sides of the edge
      auto n         = normal((v1 - v0) / length(v1 - v0));
      auto color_in  = raytrace(mesh, p - 1e-3f * n);
      auto color_out = raytrace(mesh, p + 1e-3f * n);
      // get corresponding adjoint from the adjoint image,
      // multiply with the color difference and divide by the pdf & number of samples.
      float pdf    = pmf  / (length(v1 - v0));
      float weight = 1.0f / (pdf * float(num_edge_samples));
      float adj    = dot(color_in - color_out, adjoint.color[yi * adjoint.width + xi]);
      // the boundary point is p = v0 + t * (v1 - v0)
      // according to Reynolds transport theorem, the derivatives w.r.t. q is color_diff * dot(n, dp/dq)
      // dp/dv0.x = (1 - t, 0), dp/dv0.y = (0, 1 - t)
      // dp/dv1.x = (    t, 0), dp/dv1.y = (0,     t)
      auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
      auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;
      
      // if running in parallel, use atomic add here.
      d_vertices[edge.v0] += d_v0;
      d_vertices[edge.v1] += d_v1;

      //if(mesh.type == TRIANGLE_2D_VERT_COL && d_colors!= nullptr) //#TODO: add branch by mesh.m_type
      //{
      //  // auto c0 = mesh.vertices[edge.v0];
      //  // auto c1 = mesh.vertices[edge.v1];
      //  // auto c  = c0 + t * (c1 - c0);
      //  // dc/dv0.x = (1 - t, 0), dc/dv0.y = (0, 1 - t)
      //  // dc/dv1.x = (    t, 0), dc/dv1.y = (0,     t)
      //  auto d_c0 = float3(1 - t) * adj * weight;             // * ????
      //  auto d_c1 = float3(t)     * adj * weight;             // * ????
      //  d_colors[edge.v0] += d_c0;
      //  d_colors[edge.v1] += d_c1;
      //}

      if(screen_dx != nullptr && screen_dy != nullptr) 
      {
        // for the derivatives w.r.t. p, dp/dp.x = (1, 0) and dp/dp.y = (0, 1)
        // the screen space derivatives are the negation of this
        auto dx = -n.x * (color_in - color_out) * weight;
        auto dy = -n.y * (color_in - color_out) * weight;
        // scatter gradients to buffers, in the parallel case, use atomic add here.
        screen_dx->color[yi * adjoint.width + xi] += dx;
        screen_dy->color[yi * adjoint.width + xi] += dy;
      }
    }    
}

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              mt19937 &rng,
              Img* screen_dx,
              Img* screen_dy,
              DTriangleMesh &d_mesh) {
    
  // (1)
  //
  compute_interior_derivatives(mesh, interior_samples_per_pixel, adjoint, rng, 
                               d_mesh.colors());
    
  // (2)
  //
  auto edges        = collect_edges(mesh);
  auto edge_sampler = build_edge_sampler(mesh, edges);
  compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, edge_samples_in_total,
                           rng, screen_dx, screen_dy, d_mesh.vertices());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PrintMesh(const DTriangleMesh& a_mesh)
{
  for(int i=0; i<a_mesh.numVerts();i++)
    std::cout << "ver[" << i << "]: " << a_mesh.vertices()[i].x << ", " << a_mesh.vertices()[i].y << std::endl;  
  std::cout << std::endl;
  for(size_t i=0; i<a_mesh.numFaces();i++)
    std::cout << "col[" << i << "]: " << a_mesh.colors()[i].x << ", " << a_mesh.colors()[i].y << ", " << a_mesh.colors()[i].z << std::endl;
  std::cout << std::endl;
}

float3 accumDiff3(const Img& b, const Img& a)
{
  assert(a.width*a.height == b.width*b.height);
  double accum[3] = {0,0,0};
  const size_t imgSize = a.width*a.height;
  for(size_t i=0;i<imgSize;i++)
  {
    accum[0] += double(b.color[i].x - a.color[i].x);
    accum[1] += double(b.color[i].y - a.color[i].y);
    accum[2] += double(b.color[i].z - a.color[i].z);
  }
  return float3(accum[0], accum[1], accum[2]);
}

float accumDiff(const Img& b, const Img& a)
{
  assert(a.width*a.height == b.width*b.height);
  double accum = 0.0f;
  const size_t imgSize = a.width*a.height;
  for(size_t i=0;i<imgSize;i++)
  {
    const float3 diffVec = b.color[i] - a.color[i];
    accum += double(diffVec.x + diffVec.y + diffVec.z);
  }
  return float(accum);
}

float MSEAndDiff(const Img& b, const Img& a, Img& a_outDiff)
{
  assert(a.width*a.height == b.width*b.height);
  double accumMSE = 0.0f;
  const size_t imgSize = a.width*a.height;
  for(size_t i=0;i<imgSize;i++)
  {
    const float3 diffVec = b.color[i] - a.color[i];
    a_outDiff.color[i] = 2.0f*diffVec;                     // (I[x,y] - I_target[x,y])  // dirrerential of the loss function 
    accumMSE += double(dot(diffVec, diffVec));             // (I[x,y] - I_target[x,y])^2  // loss function
  }
  return float(accumMSE);
}

float MSE(const Img& b, const Img& a)
{
  assert(a.width*a.height == b.width*b.height);
  double accumMSE = 0.0f;
  const size_t imgSize = a.width*a.height;
  for(size_t i=0;i<imgSize;i++)
  {
    const float3 diffVec = b.color[i] - a.color[i];
    accumMSE += double(dot(diffVec, diffVec));             // (I[x,y] - I_target[x,y])^2  // loss function
  }
  return float(accumMSE);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void gradFinDiff(const TriangleMesh &mesh, const char* outFolder, int width, int height,
                 DTriangleMesh &d_mesh) 
{
  Img img(width, height);
  Img ref(width, height);
  mt19937 rng(1234);
  
  ref.clear();
  render(mesh, 4, rng, ref);

  d_mesh.resize(mesh.vertices.size(), mesh.indices.size()/3, mesh.type);
  d_mesh.clear();
  
  constexpr float dPos = 0.5f;

  for(size_t i=0; i<mesh.vertices.size();i++)
  {
    TriangleMesh copy;
    
    // dx
    //
    copy = mesh;
    copy.vertices[i].x += dPos;
    img.clear();
    render(copy, 4, rng, img);

    auto diffImage   = (img - ref)/dPos;    // auto diffToTarget = (MSE(img,a_targetImage) - MSE(ref, a_targetImage))/dPos;
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "posx_" << i << ".bmp";
      auto path = strOut.str();
      save_img(diffImage, path.c_str());
    }
    float3 summColor = diffImage.summPixels(); 
    d_mesh.vertices()[i].x += 0.33333f*(summColor.x + summColor.y + summColor.z);
    
    // dy
    //
    copy = mesh;
    copy.vertices[i].y += dPos;
    img.clear();
    render(copy, 4, rng, img);

    diffImage   = (img - ref)/dPos;    // auto diffToTarget = (MSE(img,a_targetImage) - MSE(ref, a_targetImage))/dPos;
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "posy_" << i << ".bmp";
      auto path = strOut.str();
      save_img(diffImage, path.c_str());
    }
    summColor = diffImage.summPixels(); 
    d_mesh.vertices()[i].y += 0.33333f*(summColor.x + summColor.y + summColor.z);
  }

  if(mesh.type == TRIANGLE_2D_VERT_COL)
  {

  }
  else // TRIANGLE_2D_FACE_COL
  {

  }
  // for d_render use
  //Img adjoint(img.width, img.height, Vec3f{1, 1, 1});

}

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

  mt19937 rng(1234);
  Img img(256, 256);

  TriangleMesh initialMesh, targetMesh;
  scn01_TwoTrisFlat(initialMesh, targetMesh);
  //scn02_TwoTrisSmooth(initialMesh, targetMesh);
  
  if(0)
  {
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.type);
    gradFinDiff(initialMesh, "fin_diff", img.width, img.height, grad1);
    
    Img adjoint(img.width, img.height, float3{1, 1, 1});
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.type);
    d_render(initialMesh, adjoint, 4, img.width*img.height, rng, nullptr, nullptr, grad2);
    
    for(size_t i=0;i<grad1.totalParams();i++)
      std::cout << std::setprecision(4) << grad1.getData()[i] << "\t--\t" << grad2.getData()[i] << std::endl; 

    exit(0);
  }


  img.clear();
  render(targetMesh, 4 /* samples_per_pixel */, rng, img);
  save_img(img, "rendered_opt/z_target.bmp");
  
  #ifdef COMPLEX_OPT
  IOptimizer* pOpt = CreateComplexOptimizer();
  #else
  IOptimizer* pOpt = CreateSimpleOptimizer();
  #endif

  pOpt->Init(initialMesh, img); // set different target image

  TriangleMesh mesh3 = pOpt->Run(300);
  img.clear();
  render(mesh3, 4 /* samples_per_pixel */, rng, img);
  save_img(img, "rendered_opt/z_target2.bmp");
  
  delete pOpt; pOpt = nullptr;
  return 0;
}
