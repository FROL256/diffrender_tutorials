#pragma once

#include <vector>
#include <string>
#include <cstring> // for memset
#include <random>  // for random gen 
#include <cassert> 

#include "LiteMath.h"
#include "Image2d.h"

enum MESH_TYPES {
    TRIANGLE_2D_FACE_COL = 1,
    TRIANGLE_2D_VERT_COL = 2,
    TRIANGLE_2D_DIFF_TEX = 3,
    
    TRIANGLE_3D_FACE_COL = 4,
    TRIANGLE_3D_VERT_COL = 5,
    TRIANGLE_3D_DIFF_TEX = 6,
};

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;

// data structures for rendering
struct TriangleMesh 
{
  std::vector<float2>     vertices;
  std::vector<unsigned>   indices;
  std::vector<float3>     colors; // defined for each face

  MESH_TYPES type = TRIANGLE_2D_FACE_COL;
};

typedef float GradReal;

struct DTriangleMesh 
{
  DTriangleMesh(int num_vertices, int num_faces, MESH_TYPES a_meshType = TRIANGLE_2D_FACE_COL) 
  {
    resize(num_vertices, num_faces, a_meshType);
  }

  void resize(int num_vertices, int num_faces, MESH_TYPES a_meshType)
  {
    m_type = a_meshType;
    m_numVertices = num_vertices;
    m_numFaces    = num_faces;  

    if(m_type == TRIANGLE_2D_FACE_COL)
    {
      m_allParams.resize(num_vertices*2 + num_faces*3);
      m_colorOffset = num_vertices*2;
    }
    else if(m_type == TRIANGLE_2D_VERT_COL)
    {
      m_allParams.resize(num_vertices*2 + num_vertices*3);
      m_colorOffset = num_vertices*2;
    }
  }

  int numVerts() const { return m_numVertices; }
  int numFaces() const { return m_numFaces;    }
  
  //////////////////////////////////////////////////////////////////////////////////
  GradReal*       vertices_s()       { return m_allParams.data(); }
  const GradReal* vertices_s() const { return m_allParams.data(); }

  GradReal*       colors_s()       { return (m_allParams.data() + m_colorOffset); }
  const GradReal* colors_s() const { return (m_allParams.data() + m_colorOffset); }

  float2 vert_at(int i)  const { return float2(float(vertices_s()[2*i+0]), float(vertices_s()[2*i+1])); }
  float3 color_at(int i) const { return float3(float(colors_s()[3*i+0]), float(colors_s()[3*i+1]), float(colors_s()[3*i+2])); }

  //////////////////////////////////////////////////////////////////////////////////

  void clear() { for(auto& x : m_allParams) x = GradReal(0); }
  size_t totalParams() const { return m_allParams.size(); } 

  const MESH_TYPES getMeshType() const { return m_type; }
  MESH_TYPES m_type = TRIANGLE_2D_FACE_COL;

  inline const GradReal* getData() const { return m_allParams.data(); }
  inline GradReal*       getData()       { return m_allParams.data(); }

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

void render(const TriangleMesh &mesh,
            int samples_per_pixel,
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

