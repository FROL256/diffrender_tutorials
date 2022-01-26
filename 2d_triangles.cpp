#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>

#include "Bitmap.h"
#include "LiteMath.h"

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

#include <cassert>

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

// data structures for rendering
struct TriangleMesh {
    vector<float2>     vertices;
    vector<unsigned>   indices;
    vector<float3>     colors; // defined for each face
};

struct DTriangleMesh {
    DTriangleMesh(int num_vertices, int num_colors) {
        vertices.resize(num_vertices, float2{0, 0});
        colors.resize(num_colors, float3{0, 0, 0});
    }

    vector<float2> vertices;
    vector<float3> colors;
};

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

struct Img {
    Img(int width, int height, const float3 &val = float3{0, 0, 0}) :
            width(width), height(height) {
        color.resize(width * height, val);
    }

    void clear() { memset(color.data(), 0, color.size()*sizeof(float3)); }

    vector<float3> color;
    int width;
    int height;
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

void save_img(const Img &img, const string &filename, bool flip = false) 
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

// trace a single ray at screen_pos, intersect with the triangle mesh.
float3 raytrace(const TriangleMesh &mesh, const float2 &screen_pos,
                unsigned *out_faceIndex = nullptr) {
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
      
      auto side01 = dot(screen_pos - v0, n01) > 0;
      auto side12 = dot(screen_pos - v1, n12) > 0;
      auto side20 = dot(screen_pos - v2, n20) > 0;

      if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) {
          if (out_faceIndex != nullptr) {
              *out_faceIndex = i/3; // because we store face id here
          }
          return mesh.colors[i/3];
      }
    }
    // return background
    if (out_faceIndex != nullptr) {
        *out_faceIndex = unsigned(-1);
    }
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
                    auto color = raytrace(mesh, screen_pos);
                    img.color[y * img.width + x] += color / samples_per_pixel;
                }
            }
        }
    }
}

void compute_interior_derivatives(const TriangleMesh &mesh,
                                  int samples_per_pixel,
                                  const Img &adjoint,
                                  mt19937 &rng,
                                  vector<float3> &d_colors) {
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
                    raytrace(mesh, screen_pos, &faceIndex);
                    if (faceIndex != unsigned(-1)) {
                        // if running in parallel, use atomic add here.
                        d_colors[faceIndex] += adjoint.color[y * adjoint.width + x] / samples_per_pixel;
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
        Img &screen_dx,
        Img &screen_dy,
        vector<float2> &d_vertices) {
    for (int i = 0; i < num_edge_samples; i++) {
        // pick an edge
        auto edge_id = sample(edge_sampler, uni_dist(rng));
        auto edge = edges[edge_id];
        auto pmf = edge_sampler.pmf[edge_id];
        // pick a point p on the edge
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        auto t = uni_dist(rng);
        auto p = v0 + t * (v1 - v0);
        auto xi = (int)p.x; auto yi = (int)p.y; // integer coordinates
        if (xi < 0 || yi < 0 || xi >= adjoint.width || yi >= adjoint.height) {
            continue;
        }
        // sample the two sides of the edge
        auto n = normal((v1 - v0) / length(v1 - v0));
        auto color_in = raytrace(mesh, p - 1e-3f * n);
        auto color_out = raytrace(mesh, p + 1e-3f * n);
        // get corresponding adjoint from the adjoint image,
        // multiply with the color difference and divide by the pdf & number of samples.
        auto pdf = pmf / (length(v1 - v0));
        auto weight = float(1 / (pdf * float(num_edge_samples)));
        auto adj = dot(color_in - color_out, adjoint.color[yi * adjoint.width + xi]);
        // the boundary point is p = v0 + t * (v1 - v0)
        // according to Reynolds transport theorem,
        // the derivatives w.r.t. q is color_diff * dot(n, dp/dq)
        // dp/dv0.x = (1 - t, 0), dp/dv0.y = (0, 1 - t)
        // dp/dv1.x = (    t, 0), dp/dv1.y = (0,     t)
        auto d_v0 = float2{(1 - t) * n.x, (1 - t) * n.y} * adj * weight;
        auto d_v1 = float2{     t  * n.x,      t  * n.y} * adj * weight;
        // for the derivatives w.r.t. p, dp/dp.x = (1, 0) and dp/dp.y = (0, 1)
        // the screen space derivatives are the negation of this
        auto dx = -n.x * (color_in - color_out) * weight;
        auto dy = -n.y * (color_in - color_out) * weight;
        // scatter gradients to buffers
        // in the parallel case, use atomic add here.
        screen_dx.color[yi * adjoint.width + xi] += dx;
        screen_dy.color[yi * adjoint.width + xi] += dy;
        d_vertices[edge.v0] += d_v0;
        d_vertices[edge.v1] += d_v1;
    }    
}

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              mt19937 &rng,
              Img &screen_dx,
              Img &screen_dy,
              DTriangleMesh &d_mesh) {
    compute_interior_derivatives(mesh, interior_samples_per_pixel, adjoint, rng, d_mesh.colors);
    auto edges = collect_edges(mesh);
    auto edge_sampler = build_edge_sampler(mesh, edges);
    compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, edge_samples_in_total,
                             rng, screen_dx, screen_dy, d_mesh.vertices);
}

void PrintMesh(const DTriangleMesh& a_mesh)
{
  for(size_t i=0; i<a_mesh.vertices.size();i++)
    std::cout << "ver[" << i << "]: " << a_mesh.vertices[i].x << ", " << a_mesh.vertices[i].y << std::endl;  
  std::cout << std::endl;
  for(size_t i=0; i<a_mesh.colors.size();i++)
    std::cout << "col[" << i << "]: " << a_mesh.colors[i].x << ", " << a_mesh.colors[i].y << ", " << a_mesh.colors[i].z << std::endl;
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

int main(int argc, char *argv[]) 
{
  #ifdef WIN32
  mkdir("rendered");
  #else
  mkdir("rendered", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  TriangleMesh mesh{
      // vertices
      {{50.0, 25.0}, {200.0, 200.0}, {15.0, 150.0},
       {200.0, 15.0}, {150.0, 250.0}, {50.0, 100.0}},
      // indices
      {0, 1, 2, 
       3, 4, 5},
      // color
      {{0.3, 0.5, 0.3}, {0.3, 0.3, 0.5}}
  };

  Img img(256, 256);
  mt19937 rng(1234);
  render(mesh, 4 /* samples_per_pixel */, rng, img);
  save_img(img, "rendered/render.bmp");

  Img adjoint(img.width, img.height, float3{1, 1, 1});
  Img dx(img.width, img.height), dy(img.width, img.height);
  
  DTriangleMesh d_mesh(mesh.vertices.size(), mesh.colors.size());
  d_render(mesh, adjoint, 4 /* interior_samples_per_pixel */,
           img.width * img.height /* edge_samples_in_total */, rng, dx, dy, d_mesh);
  save_img(dx, "rendered/dx_pos.bmp", false /*flip*/); save_img(dx, "rendered/dx_neg.bmp", true /*flip*/);
  save_img(dy, "rendered/dy_pos.bmp", false /*flip*/); save_img(dy, "rendered/dy_neg.bmp", true /*flip*/);
  
  std::cout << "gradients(1):" << std::endl;
  PrintMesh(d_mesh);

  // check differentials with brute force numerical approach
  //
  DTriangleMesh testMesh(mesh.vertices.size(), mesh.colors.size());
  Img           tempImg(256, 256); 
  const float   dEpsilon = 2.0f;

  for(size_t vertId=0; vertId< mesh.vertices.size(); vertId++)
  {
    TriangleMesh tmpMesh = mesh;
    tmpMesh.vertices[vertId].x += dEpsilon;
    tempImg.clear();
    render(tmpMesh, 4 /* samples_per_pixel */, rng, tempImg);
    testMesh.vertices[vertId].x += accumDiff(tempImg, img)/dEpsilon;
  
    tmpMesh = mesh;
    tmpMesh.vertices[vertId].y += dEpsilon;
    tempImg.clear();
    render(tmpMesh, 4 /* samples_per_pixel */, rng, tempImg);
    testMesh.vertices[vertId].y += accumDiff(tempImg, img)/dEpsilon;
  }
  
  for(size_t faceId=0; faceId < mesh.colors.size(); faceId++)
  {
    TriangleMesh tmpMesh = mesh;
    tmpMesh.colors[faceId] += float3(dEpsilon, dEpsilon, dEpsilon);
    tempImg.clear();
    render(tmpMesh, 4 /* samples_per_pixel */, rng, tempImg);
    testMesh.colors[faceId] += accumDiff3(tempImg, img)/dEpsilon;
  }

  std::cout << std::endl;
  std::cout << "gradients(2):" << std::endl;
  PrintMesh(testMesh);

  return 0;
}
