#pragma once

#include <vector>
#include <string>
#include <cstring> // for memset
#include <random>  // for random gen 

#include "LiteMath.h"

// data structures for rendering
struct TriangleMesh {
    std::vector<LiteMath::float2>     vertices;
    std::vector<unsigned>             indices;
    std::vector<LiteMath::float3>     colors; // defined for each face
};

struct DTriangleMesh {
    DTriangleMesh(int num_vertices, int num_colors) {
        vertices.resize(num_vertices, LiteMath::float2{0, 0});
        colors.resize(num_colors, LiteMath::float3{0, 0, 0});
    }

    std::vector<LiteMath::float2> vertices;
    std::vector<LiteMath::float3> colors;
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
