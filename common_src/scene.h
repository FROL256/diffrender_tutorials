#pragma once
#include <vector>
#include <string>
#include <cstring> 
#include <cassert> 
#include <map>
#include "utils.h"

class IOptimizer;
class OptSimple;
class Tester;

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

/**
\brief input/output mesh
*/
struct TriangleMesh 
{
  TriangleMesh() = default;
  TriangleMesh(const std::vector<float3> &_vertices, const std::vector<float3> &_colors, const std::vector<unsigned> &_indices = {})
  {
    vertices = _vertices;
    colors = _colors;
    indices = _indices;
  }
  
  TriangleMesh(const std::vector<float3> &_vertices, const std::vector<float2> &_tc, const std::vector<unsigned> &_indices = {})
  {
    vertices = _vertices;
    tc = _tc;
    indices = _indices;
  }

  inline int vertex_count() const { return vertices.size(); }
  inline int face_count() const { return indices.size()/3; }

  //vertex attributes, some of them might be empty
  std::vector<float3>     vertices;
  std::vector<float3>     colors;
  std::vector<float2>     tc;
  std::vector<float3>     normals;
  std::vector<float3>     tangents;

  std::vector<unsigned>   indices;

  std::vector<CPUTexture> textures; // an arbitrary number of textures
};

struct PointLight
{
  PointLight() = default;
  PointLight(const float3 &col, float inten, const float3 &_pos)
  {
    color = normalize(col);
    intensity = inten;
    pos = _pos;
  }

  float3 color;
  float intensity;
  float3 pos;
};

struct AreaLight
{
  AreaLight() = default;
  AreaLight(const TriangleMesh &_mesh, const float3 &col, float inten, const float3 &_pos)
  {
    color = normalize(col);
    intensity = inten;
    pos = _pos;
    mesh = _mesh;
  }

  float3 color;
  float intensity;
  float3 pos;
  TriangleMesh mesh;
};

/**
\brief base scene description without auxilary data for differentiable rendering
*/
struct Scene
{
public:
  void add_mesh(const TriangleMesh &mesh, std::string name = "")
  {
    meshes.push_back(mesh);
    meshes_by_name.emplace(name, meshes.size()-1);
  }

  void set_mesh(const TriangleMesh &mesh, int id)
  {
    if (id >= meshes.size())
      meshes.resize(id+1);
    meshes[id] = mesh;
  }

  inline unsigned get_index(unsigned n) const { return meshes[0].indices[n]; }//TODO: support multiple meshes
  inline float3 get_pos(unsigned id)    const { return meshes[0].vertices[id]; }//TODO: support multiple meshes
  inline float3 get_color(unsigned id)  const { return meshes[0].colors[id]; }//TODO: support multiple meshes
  inline float3 get_norm(unsigned id)   const { return meshes[0].normals[id]; }//TODO: support multiple meshes
  inline float3 get_tang(unsigned id)   const { return meshes[0].tangents[id]; }//TODO: support multiple meshes
  inline float2 get_tc(unsigned id)     const { return meshes[0].tc[id]; }//TODO: support multiple meshes
  inline const CPUTexture &get_tex(unsigned n) const { return meshes[0].textures[n]; }//TODO: support multiple meshes
  inline const TriangleMesh &get_mesh(unsigned n) const { return meshes[n]; }
  inline const std::vector<TriangleMesh> &get_meshes() const { return meshes; }
  inline unsigned indices_size() const 
  {
    unsigned c = 0;
    for (auto &m : meshes)
      c+=m.indices.size();
    return c;
  }

  friend class IOptimizer;
  friend class OptSimple;
  friend class Tester;

protected:
  std::vector<TriangleMesh> meshes;
  std::map<std::string, int> meshes_by_name; //position in meshes vector

  float3 ambient_light_color = float3(0,0,0);
  float3 environment_light_mult = float3(1,1,1);
  CPUTexture environment_light_texture;
  std::vector<PointLight> point_lights;
  std::vector<AreaLight> area_lights;

};
