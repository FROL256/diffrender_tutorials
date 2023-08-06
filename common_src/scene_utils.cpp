#include "scene.h"

void Scene::transform_meshes(bool restore_normals, bool restore_tangents, bool transform_to_unindexed_mesh)
{
  for (auto &mesh : meshes)
  {
    if (restore_normals && mesh.normals.size() != mesh.vertices.size())
    {
      mesh.normals = std::vector<float3>(mesh.vertices.size(), float3(0,0,0));
      std::vector<int> vert_mul(mesh.vertices.size(), 0);

      for (int i=0;i<mesh.indices.size(); i+=3)
      {
        float3 &v1 = mesh.vertices[mesh.indices[i]];
        float3 &v2 = mesh.vertices[mesh.indices[i+1]];
        float3 &v3 = mesh.vertices[mesh.indices[i+2]];

        float3 l1 = v2-v1;
        float3 l2 = v3-v1;
        float3 n = float3(1,0,0);
        if (length(l1) < 1e-6 || length(l2) < 1e-6)
        {
          logerr("Scene::transform_meshes triangle[%d %d %d] has near-zero size. It may lead to errors",
                 mesh.indices[i], mesh.indices[i+1], mesh.indices[i+2]);
          n = normalize(cross(l1, l2));
        }
        mesh.normals[mesh.indices[i]] += n;
        mesh.normals[mesh.indices[i+1]] += n;
        mesh.normals[mesh.indices[i+2]] += n;

        vert_mul[mesh.indices[i]] += 1;
        vert_mul[mesh.indices[i+1]] += 1;
        vert_mul[mesh.indices[i+2]] += 1;
      }

      for (int i=0; i<mesh.normals.size(); i++)
        mesh.normals[i] /= vert_mul[i];
    }

    if (restore_tangents)
    {

    }

    if (transform_to_unindexed_mesh)
    {
      if (mesh.colors.size() != mesh.vertices.size())
        mesh.colors = std::vector<float3>(mesh.vertices.size(), float3(0,0,0));
      if (mesh.tc.size() != mesh.vertices.size())
        mesh.tc = std::vector<float2>(mesh.vertices.size(), float2(0,0));
      if (mesh.normals.size() != mesh.vertices.size())
        mesh.normals = std::vector<float3>(mesh.vertices.size(), float3(1,0,0));
      if (mesh.tangents.size() != mesh.vertices.size())
        mesh.tangents = std::vector<float3>(mesh.vertices.size(), float3(0,1,0));
    
      int v_count = mesh.indices.size();
      auto vertices = std::vector<float3>(v_count, float3(0,0,0));
      auto colors = std::vector<float3>(v_count, float3(0,0,0));
      auto tc = std::vector<float2>(v_count, float2(0,0));
      auto normals = std::vector<float3>(v_count, float3(0,0,0));
      auto tangents = std::vector<float3>(v_count, float3(0,0,0));
      auto indices = std::vector<unsigned int>(v_count, 0);

      int i = 0;
      for (int ind : mesh.indices)
      {
        vertices[i] = mesh.vertices[ind];
        colors[i] = mesh.colors[ind];
        tc[i] = mesh.tc[ind];
        normals[i] = mesh.normals[ind];
        tangents[i] = mesh.tangents[ind];
        indices[i] = i;
        i++;
      }

      mesh.vertices = vertices;
      mesh.colors = colors;
      mesh.tc = tc;
      mesh.normals = normals;
      mesh.tangents = tangents;
      mesh.indices = indices;
    }
  }
}