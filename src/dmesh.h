#pragma once

#include <vector>
#include <string>
#include <cstring> // for memset
#include <random>  // for random gen 
#include <cassert> 

#include "LiteMath.h"
#include "Image2d.h"
#include "utils.h"
#include "scene.h"

enum class SHADING_MODEL {UNDEFINED = 0,
                          SILHOUETTE = 1,
                          VERTEX_COLOR = 2,
                          DIFFUSE = 3, 
                          LAMBERT = 4,
                          PHONG = 5,
                          GGX = 6,
                          PATH_TEST = 7};

typedef float GradReal;

/**
\brief gradient of mesh

*/
struct DTriangleMesh 
{
  DTriangleMesh(){}
  DTriangleMesh(const TriangleMesh &mesh, SHADING_MODEL material) { reset(mesh, material); }

  void reset(const TriangleMesh &mesh, SHADING_MODEL material)
  {
    m_numVertices = mesh.vertex_count();
    m_numFaces    = mesh.face_count();  
    
    if (material == SHADING_MODEL::VERTEX_COLOR)
      m_allParams.resize((3 + 3)*m_numVertices);
    else if (material == SHADING_MODEL::SILHOUETTE)
      m_allParams.resize(3*m_numVertices);
    else if (material != SHADING_MODEL::UNDEFINED)
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
