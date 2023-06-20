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

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;

using LiteMath::int2;
using LiteMath::int3;
using LiteMath::int4;

struct CPUTexture
{
  CPUTexture() = default;
  CPUTexture(const LiteImage::Image2D<float3> &img)
  {
    w = img.width();
    h = img.height();
    channels = 3;
    data = std::vector<float>((float*)img.data(), (float*)img.data() + w*h*channels);
  }

  inline int pixel_to_offset(int x, int y) const { return channels*(y*w + x); }
  inline int pixel_to_offset(int2 pixel) const { return channels*(pixel.y*w + pixel.x); }
  /**
  \brief UNSAFE ACCESS!!!

  */
  const float *get(int x, int y) const
  {
    return data.data() + pixel_to_offset(x,y); 
  }
  std::vector<float> data;
  int w,h,channels;
};

enum class MATERIAL { UNDEFINED = 0,
                      DIFFUSE = 1, 
                      LAMBERT = 2};

/**
\brief input/output mesh

*/
struct TriangleMesh 
{
  std::vector<float3>     vertices;
  std::vector<float3>     colors; // defined for each face
  std::vector<unsigned>   indices;

  std::vector<float2>     tc; //if m_meshType != TRIANGLE_DIFF_TEX vector is empty
  MATERIAL material = MATERIAL::UNDEFINED;
  std::vector<CPUTexture> textures; // an arbitrary number of textures

  MESH_TYPES m_meshType = MESH_TYPES::TRIANGLE_FACE_COL;
};

typedef float GradReal;

/**
\brief gradient of mesh

*/
struct DTriangleMesh 
{
  DTriangleMesh(){}

  void reset(const TriangleMesh &mesh)
  {
    m_meshType = mesh.m_meshType;
    m_numVertices = mesh.vertices.size();
    m_numFaces    = mesh.indices.size()/3;  
    
    if(m_meshType == MESH_TYPES::TRIANGLE_VERT_COL)
      m_allParams.resize(m_numVertices*3 + m_numVertices*3);
    else if(m_meshType == MESH_TYPES::TRIANGLE_FACE_COL)
      m_allParams.resize(m_numVertices*3 + m_numFaces*3);
    else if (m_meshType == MESH_TYPES::TRIANGLE_DIFF_TEX)
    {
      int off = m_numVertices*3;
      for (auto &t : mesh.textures)
      {
        m_texOffsets.push_back(off);
        off += t.w*t.h*t.channels;
      }
      m_allParams.resize(off);
    }
    else
      assert(false);
    m_colorOffset = m_numVertices*3;

    clear();
  }

  int numVerts() const { return m_numVertices; }
  int numFaces() const { return m_numFaces;    }

  //////////////////////////////////////////////////////////////////////////////////
  int vert_offs () const { return 0; }
  int color_offs() const { return m_colorOffset; }

  GradReal*       vertices_s()       { return m_allParams.data() + vert_offs(); }
  const GradReal* vertices_s() const { return m_allParams.data() + vert_offs(); }

  GradReal*       colors_s()       { return (m_allParams.data() + color_offs()); }
  const GradReal* colors_s() const { return (m_allParams.data() + color_offs()); }

  float3 vert_at(int i)  const { return float3(float(vertices_s()[3*i+0]), float(vertices_s()[3*i+1]), float(vertices_s()[3*i+2])); }
  float3 color_at(int i) const { return float3(float(colors_s  ()[3*i+0]), float(colors_s  ()[3*i+1]), float(colors_s  ()[3*i+2])); }

  std::vector<GradReal> subvecPos() const { return std::vector<GradReal>(m_allParams.begin(), m_allParams.begin() + color_offs()); }
  std::vector<GradReal> subvecCol() const { return std::vector<GradReal>(m_allParams.begin() + color_offs(), m_allParams.end()); }

  //////////////////////////////////////////////////////////////////////////////////

  void clear() { for(auto& x : m_allParams) x = GradReal(0); }
  size_t size() const { return m_allParams.size(); } 

  MESH_TYPES getMeshType() const { return m_meshType; }

  MESH_TYPES m_meshType = MESH_TYPES::TRIANGLE_FACE_COL;

  inline const GradReal* data() const { return m_allParams.data(); }
  inline GradReal*       data()       { return m_allParams.data(); }

  inline GradReal& operator[](size_t i)       { return m_allParams[i]; }
  inline GradReal  operator[](size_t i) const { return m_allParams[i]; }
  
  inline int tex_count() const { return m_texOffsets.size(); }
  inline int tex_offset(int tex_n) const { return m_texOffsets[tex_n]; }

protected:

  std::vector<GradReal> m_allParams;
  int m_colorOffset;
  int m_numVertices;
  int m_numFaces;

  std::vector<int> m_texOffsets;
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


static inline float2 normal2D(const float2 &v) {return float2{-v.y, v.x};} 
static inline float  edgeFunction(float2 a, float2 b, float2 c) { return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x); }  // actuattly just a mixed product ... :)
