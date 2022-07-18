#pragma once

#include <vector>
#include <string>
#include <cstring> // for memset
#include <random>  // for random gen 
#include <cassert> 

#include "LiteMath.h"
#include "Image2d.h"

enum class MESH_TYPES {
    TRIANGLE_FACE_COL = 1,
    TRIANGLE_VERT_COL = 2,
    TRIANGLE_DIFF_TEX = 3,
};

enum class GEOM_TYPE { TRIANGLE_2D = 0, 
                       TRIANGLE_3D = 1};

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;

// data structures for rendering
struct TriangleMesh 
{
  std::vector<float3>     vertices;
  std::vector<float3>     colors; // defined for each face
  std::vector<unsigned>   indices;

  MESH_TYPES m_meshType = MESH_TYPES::TRIANGLE_FACE_COL;
  GEOM_TYPE  m_geomType = GEOM_TYPE::TRIANGLE_2D;
};

typedef float GradReal;

struct DTriangleMesh 
{
  DTriangleMesh(int num_vertices, int num_faces, MESH_TYPES a_meshType = MESH_TYPES::TRIANGLE_FACE_COL, GEOM_TYPE a_gType = GEOM_TYPE::TRIANGLE_2D) 
  {
    m_meshType = a_meshType;
    m_geomType = a_gType;
    resize(num_vertices, num_faces);
  }

  void resize(int num_vertices, int num_faces)
  {
    m_numVertices = num_vertices;
    m_numFaces    = num_faces;  
    
    if(m_meshType == MESH_TYPES::TRIANGLE_VERT_COL)
      m_allParams.resize(num_vertices*3 + num_vertices*3);
    else
      m_allParams.resize(num_vertices*3 + num_faces*3);
  
    m_colorOffset = num_vertices*3;
  }

  int numVerts() const { return m_numVertices; }
  int numFaces() const { return m_numFaces;    }

  //////////////////////////////////////////////////////////////////////////////////
  GradReal*       vertices_s()       { return m_allParams.data(); }
  const GradReal* vertices_s() const { return m_allParams.data(); }

  GradReal*       colors_s()       { return (m_allParams.data() + m_colorOffset); }
  const GradReal* colors_s() const { return (m_allParams.data() + m_colorOffset); }

  float3 vert_at(int i)  const { return float3(float(vertices_s()[3*i+0]), float(vertices_s()[3*i+1]), float(vertices_s()[3*i+2])); }
  float3 color_at(int i) const { return float3(float(colors_s  ()[3*i+0]), float(colors_s  ()[3*i+1]), float(colors_s  ()[3*i+2])); }

  std::vector<GradReal> subvecPos() const { return std::vector<GradReal>(m_allParams.begin(), m_allParams.begin() + m_colorOffset); }
  std::vector<GradReal> subvecCol() const { return std::vector<GradReal>(m_allParams.begin() + m_colorOffset, m_allParams.end()); }

  //////////////////////////////////////////////////////////////////////////////////

  void clear() { for(auto& x : m_allParams) x = GradReal(0); }
  size_t totalParams() const { return m_allParams.size(); } 

  const MESH_TYPES getMeshType() const { return m_meshType; }
  MESH_TYPES m_meshType = MESH_TYPES::TRIANGLE_FACE_COL;
  GEOM_TYPE  m_geomType = GEOM_TYPE::TRIANGLE_2D;

  inline const GradReal* getData() const { return m_allParams.data(); }
  inline GradReal*       getData()       { return m_allParams.data(); }

  inline float& operator[](size_t i)       { return m_allParams[i]; }
  inline float  operator[](size_t i) const { return m_allParams[i]; }

protected:

  std::vector<GradReal> m_allParams;
  int m_colorOffset;
  int m_numVertices;
  int m_numFaces;
};

using Img = LiteImage::Image2D<float3>;

static inline float3 SummOfPixels(const Img& a_image) 
{
  const auto& color = a_image.vector();
  double summ[3] = {0.0, 0.0, 0.0};
  for(size_t i=0;i<color.size();i++) {
    summ[0] += double(color[i].x);
    summ[1] += double(color[i].y);
    summ[2] += double(color[i].z); 
  }
  return float3(summ[0], summ[1], summ[2]);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float LossAndDiffLoss(const Img& b, const Img& a, Img& a_outDiff);

void render(const TriangleMesh &mesh, int samples_per_pixel,
            Img &img) ;

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              Img* screen_dx,
              Img* screen_dy,
              DTriangleMesh &d_mesh);

void opt_step(const DTriangleMesh &gradMesh, float alphaPos, float alphaColor, 
              TriangleMesh *mesh);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float2 normal2D(const float2 &v) {return float2{-v.y, v.x};} 
static inline float  edgeFunction(float2 a, float2 b, float2 c) { return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x); }  // actuattly just a mixed product ... :)
