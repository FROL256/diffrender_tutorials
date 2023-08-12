#pragma once
#include <vector>
#include <string>
#include <cstring> 
#include <cassert> 
#include <map>
#include "utils.h"

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

void transform(TriangleMesh &mesh, const LiteMath::float4x4 &transform);

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
  void add_mesh(const TriangleMesh &mesh, const std::vector<float4x4> &transform = {float4x4()}, std::string name = "")
  {
    meshes.push_back(mesh);
    transforms.push_back(transform);
    meshes_by_name.emplace(name, meshes.size()-1);
    prepared = false;
  }

  void set_mesh(const TriangleMesh &mesh, int id, const std::vector<float4x4> &transform = {float4x4()})
  {
    if (id >= meshes.size())
      meshes.resize(id+1);
    if (id >= transforms.size())
      transforms.resize(id+1);
    meshes[id] = mesh;
    transforms[id] = transform;
    prepared = false;
  }

  inline unsigned get_index(unsigned mesh_id, unsigned instance_id, unsigned vertex_id) const 
  { 
    return preparedData.indices[mesh_id][instance_id][vertex_id]; 
  }
  //id is an index returned by get_index() function
  inline float3 get_pos(unsigned id)     const { return preparedData.vertices[id]; }
  inline float3 get_pos_orig(unsigned id)const { return preparedData.orig_vertices[id]; }
  inline float3 get_color(unsigned id)   const { return preparedData.colors[id]; }
  inline float3 get_norm(unsigned id)    const { return preparedData.normals[id]; }
  inline float3 get_tang(unsigned id)    const { return preparedData.tangents[id]; }
  inline float2 get_tc(unsigned id)      const { return preparedData.tc[id]; }
  
  inline const CPUTexture &get_tex(unsigned n) const { return meshes[0].textures[n]; }//TODO: support multiple meshes
  
  inline const TriangleMesh &get_mesh(unsigned n) const { return meshes[n]; }
  inline const std::vector<TriangleMesh> &get_meshes() const { return meshes; }
  inline TriangleMesh &get_mesh_modify(unsigned n) { prepared = false; return meshes[n]; }
  inline std::vector<TriangleMesh> &get_meshes_modify() { prepared = false; return meshes; }
  
  inline const std::vector<float4x4>  &get_transform(unsigned n) const { return transforms[n]; }
  inline const std::vector<std::vector<float4x4>>  &get_transforms() const { return transforms; }
  inline const std::vector<float4x4>  &get_transform_inv(unsigned n) const { return transforms_inv[n]; }
  inline const std::vector<std::vector<float4x4>>  &get_transforms_inv() const { return transforms_inv; }
  inline std::vector<float4x4>  &get_transform_modify(unsigned n) { prepared = false; return transforms[n];  }
  inline std::vector<std::vector<float4x4>>  &get_transforms_modify() { prepared = false; return transforms; }
  inline unsigned indices_size() const 
  {
    unsigned c = 0;
    for (auto &m : meshes)
      c+=m.indices.size();
    return c;
  }

  void restore_meshes(bool restore_normals, bool restore_tangents, bool transform_to_unindexed_mesh);

  //applies all transorms to meshes and puts them into one big structure
  void prepare_for_render() const;

  void get_prepared_mesh(TriangleMesh &mesh) const;

protected:
  std::vector<TriangleMesh> meshes;
  std::vector<std::vector<float4x4>> transforms;
  mutable std::vector<std::vector<float4x4>> transforms_inv;
  std::map<std::string, int> meshes_by_name; //position in meshes vector

  float3 ambient_light_color = float3(0,0,0);
  float3 environment_light_mult = float3(1,1,1);
  CPUTexture environment_light_texture;
  std::vector<PointLight> point_lights;
  std::vector<AreaLight> area_lights;

  mutable bool prepared = false;
  mutable struct PreparedData
  {
    //3-dimentional array id = indices[mesh_id][instance_id][vertex_id] 
    std::vector<std::vector<std::vector<unsigned>>> indices;

    std::vector<float3>     vertices;
    std::vector<float3>     orig_vertices;
    std::vector<float3>     colors;
    std::vector<float2>     tc;
    std::vector<float3>     normals;
    std::vector<float3>     tangents;
  } preparedData;
};
