#include "scenes.h"

void scn01_TwoTrisFlat(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh mesh{
      // vertices
      {{50.0, 25.0}, {200.0, 200.0}, {15.0, 150.0},
       {200.0, 15.0}, {150.0, 250.0}, {50.0, 100.0}},
      // indices
      {0, 1, 2, 
       3, 4, 5},
      // color
      {{0.3, 0.3, 0.3}, 
      {0.3, 0.3, 0.3}}
  };

  initial = mesh;

  TriangleMesh mesh2{
      // vertices
      {{50.0, 25.0+10.0}, {200.0, 200.0+10.0}, {15.0, 150.0+10.0},
       {200.0-10.0 + 50.0, 15.0+5.0}, {150.0+50.0+50.0, 250.0-25.0}, {80.0, 100.0-25.0}},
      // indices
      {0, 1, 2, 
       3, 4, 5},
      // color
      {{0.3, 0.5, 0.3}, {0.3, 0.3, 0.5}}
  };
  target = mesh2;
}

void scn02_TwoTrisSmooth(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh mesh{
      // vertices
      {{50.0, 25.0}, {200.0, 200.0}, {15.0, 150.0},
       {200.0, 15.0}, {150.0, 250.0}, {50.0, 100.0}},
      // indices
      {0, 1, 2, 
       3, 4, 5}
  };
  
  mesh.type = TRIANGLE_2D_VERT_COL;
  mesh.colors.resize(mesh.vertices.size());
  for(size_t i=0;i<mesh.vertices.size();i++)
     mesh.colors[i] = float3{0.3, 0.5, 0.3};
  initial = mesh;
  ///////////////////////////////////////////////////////////////// 
  
  TriangleMesh mesh2{
      // vertices
      {{50.0, 25.0+10.0}, {200.0, 200.0+10.0}, {15.0, 150.0+10.0},
       {200.0-10.0 + 50.0, 15.0+5.0}, {150.0+50.0+50.0, 250.0-25.0}, {80.0, 100.0-25.0}},
      // indices
      {0, 1, 2, 
       3, 4, 5},
  };

  mesh2.type = TRIANGLE_2D_VERT_COL;
  mesh2.colors.resize(mesh.vertices.size());

  mesh2.colors[0] = float3(1,0,0);
  mesh2.colors[1] = float3(0,1,0);
  mesh2.colors[2] = float3(0,0,1);

  mesh2.colors[3] = float3(1,1,0);
  mesh2.colors[4] = float3(1,1,0);
  mesh2.colors[5] = float3(1,1,0);

  target = mesh2;
}