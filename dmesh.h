#pragma once

#include <vector>
#include <string>
#include <cstring> // for memset
#include <random>  // for random gen 

#include "LiteMath.h"

enum MESH_TYPES {
    TRIANGLE_2D_FACE_COL = 1,
    TRIANGLE_2D_VERT_COL = 2,
    TRIANGLE_2D_DIFF_TEX = 3,
    TRIANGLE_3D_FACE_COL = 4,
    TRIANGLE_3D_VERT_COL = 5,
    TRIANGLE_3D_DIFF_TEX = 6,
};

// data structures for rendering
struct TriangleMesh {
    std::vector<LiteMath::float2>     vertices;
    std::vector<unsigned>             indices;
    std::vector<LiteMath::float3>     colors; // defined for each face
};

struct DTriangleMesh {
    DTriangleMesh(int num_vertices, int num_colors) {
      //vertices.resize(num_vertices, LiteMath::float2{0, 0});
      //colors.resize(num_colors, LiteMath::float3{0, 0, 0});

      m_numVertices = num_vertices;
      m_numFaces    = num_colors;
      m_allParams.resize(num_vertices*2 + num_colors*3);
      m_faceColorOffset = num_vertices*2;
    }

    //std::vector<LiteMath::float2> vertices;
    //std::vector<LiteMath::float3> colors;

    int numVertices() const { return m_numVertices; }
    int numFaces()    const { return m_numFaces;    }
 
    LiteMath::float2*       vertices()       { return (LiteMath::float2*)m_allParams.data(); }
    const LiteMath::float2* vertices() const { return (LiteMath::float2*)m_allParams.data(); }

    LiteMath::float3*       faceColors()       { return (LiteMath::float3*)(m_allParams.data() + m_faceColorOffset); }
    const LiteMath::float3* faceColors() const { return (LiteMath::float3*)(m_allParams.data() + m_faceColorOffset); }

    void clear() { memset(m_allParams.data(), 0, m_allParams.size()*sizeof(float)); }
    size_t totalParams() const { return m_allParams.size(); } 

protected:

    inline const float* getData() const { return m_allParams.data(); }
    inline float*       getData()       { return m_allParams.data(); }
    std::vector<float> m_allParams;

    int                m_faceColorOffset;
    int                m_numVertices;
    int                m_numFaces;
};

struct Img {
    Img(){}
    Img(int width, int height, const LiteMath::float3 &val = LiteMath::float3{0, 0, 0}) : width(width), height(height) {
        color.resize(width * height, val);
    }

    void clear() { memset(color.data(), 0, color.size()*sizeof(LiteMath::float3)); }

    std::vector<LiteMath::float3> color;
    int width;
    int height;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float MSEAndDiff(const Img& b, const Img& a, Img& a_outDiff);

void render(const TriangleMesh &mesh,
            int samples_per_pixel,
            std::mt19937 &rng,
            Img &img) ;

void d_render(const TriangleMesh &mesh,
              const Img &adjoint,
              const int interior_samples_per_pixel,
              const int edge_samples_in_total,
              std::mt19937 &rng,
              Img &screen_dx,
              Img &screen_dy,
              DTriangleMesh &d_mesh);

void save_img(const Img &img, const std::string &filename, bool flip = false);
