#include "scenes.h"
#include <iostream>

void scn01_TwoTrisFlat(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh mesh{
      // vertices
      {{50.0, 25.0, 0.0}, {200.0, 200.0, 0.0}, {15.0, 150.0, 0.0},
       {200.0, 15.0, 0.0}, {150.0, 250.0, 0.0}, {50.0, 100.0, 0.0}},
      // color
      {{0.3, 0.3, 0.3}, {0.3, 0.3, 0.3}},
      // indices
      {0, 1, 2, 
       3, 4, 5}
  };

  initial = mesh;

  TriangleMesh mesh2{
      // vertices
      {{50.0, 25.0+10.0, 0.0}, {200.0, 200.0+10.0, 0.0}, {15.0, 150.0+10.0, 0.0},
       {200.0-10.0 + 50.0, 15.0+5.0, 0.0}, {150.0+50.0+50.0, 250.0-25.0, 0.0}, {80.0, 100.0-25.0, 0.0}},
      // color
      {{0.3, 0.5, 0.3}, {0.3, 0.3, 0.5}},
      // indices
      {0, 1, 2, 
       3, 4, 5},
  };
  target = mesh2;
}

void scn02_TwoTrisSmooth(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh mesh{
      // vertices
      {{50.0, 25.0, 0.0}, {200.0, 200.0, 0.0}, {15.0, 150.0, 0.0},
       {200.0, 15.0, 0.0}, {150.0, 250.0, 0.0}, {50.0, 100.0, 0.0}},
       
      {{0.0, 0.0, 0.75}, {0.5, 0.1, 0.0}, {0.0, 0.75, 0.5},
       {0.3, 0.3, 0.5},  {0.3, 0.5, 0.3}, {0.3, 0.5, 0.3}}, 

      // indices
      {0, 1, 2, 
       3, 4, 5}
  };
  
  mesh.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  initial = mesh;
  ///////////////////////////////////////////////////////////////// 
  
  TriangleMesh mesh2{
      // vertices
      {{50.0, 25.0+10.0, 0.0}, {200.0, 200.0+10.0, 0.0}, {15.0, 150.0+10.0, 0.0},
       {200.0-10.0 + 50.0, 15.0+5.0, 0.0}, {150.0+50.0+50.0, 250.0-25.0, 0.0}, {80.0, 100.0-25.0, 0.0}},

       {{1,0,0}, {0,1,0}, {0,0,1},
        {1,1,0}, {1,1,0}, {1,1,0}}, 

      // indices
      {0, 1, 2, 
       3, 4, 5}
  };

  mesh2.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  target = mesh2;
}

void scn03_Triangle3D   (TriangleMesh& initial, TriangleMesh& target)
{
   TriangleMesh pyramid{
      // vertices
      {{0.0f, 1.0f, 0.0f},    
       {1.0f, -1.0f, 1.0f},
       {-1.0f, -1.0f, 1.0f},  
       },

      // color
      //{{1.0f, 0.0f, 0.0f}, 
      // {1.0f, 1.0f, 0.0f}, 
      // {0.0f, 0.0f, 1.0f},
      // },
      
      //{{0.07805659f, 0.07805659f, 0.07805659f}, 
      // {0.07805659f, 0.07805659f, 0.07805659f}, 
      // {0.07805659f, 0.07805659f, 0.07805659f},
      // },
      
      //{{1.0f, 1.0f, 1.0f}, 
      // {1.0f, 1.0f, 1.0f}, 
      // {1.0f, 1.0f, 1.0f},
      // },

      {{1.0f, 0.0f, 0.0f}, 
       {0.0f, 0.0f, 0.0f}, 
       {1.0f, 0.0f, 0.0f},
       },

      // indices
      {0, 1, 2}
  };

  pyramid.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  pyramid.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  initial = pyramid;
  target  = pyramid;
  
  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.5f,-5.0f));
  LiteMath::float4x4 mRotate1   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-40.0f);
  LiteMath::float4x4 mRotate2   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*+30.0f);
  
  auto mTransform1 = mTranslate*mRotate1;
  auto mTransform2 = mTranslate*mRotate2;

  for(auto& v : initial.vertices)
    v = (mTransform1*v); // + float3(0,0,-0.01f);

  for(auto& v : target.vertices)
    v = (mTransform2*v); // + float3(0,0,-0.01f);
  
  std::cout << "initial: [" << std::endl;
  for(const auto& v : initial.vertices)
    std::cout << "[" << v[0] << ", " <<  v[1] << ", " << v[2] << "] "  << std::endl;
  std::cout << "]" << std::endl << std::endl;

  std::cout << "target: [" << std::endl;
  for(const auto& v : target.vertices)
    std::cout << "[" << v[0] << ", " <<  v[1] << ", " << v[2] << "] "  << std::endl;
  std::cout << "]" << std::endl << std::endl;
}


void scn04_Pyramid3D(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh pyramid{
      // vertices
      {{0.0f, 1.0f, 0.0f},    
       {-1.0f, -1.0f, 1.0f},  
       {1.0f, -1.0f, 1.0f},
       {1.0f, -1.0f, -1.0f},
       {-1.0f, -1.0f, -1.0f},
       },

      // color
      {{1.0f, 0.0f, 0.0f}, 
       {1.0f, 1.0f, 0.0f}, 
       {0.0f, 0.0f, 1.0f},
       {0.0f, 1.0f, 0.0f}, 
       {0.0f, 0.0f, 1.0f}, 
       },

      // indices
      {0, 1, 2,
       0, 2, 3,
       0, 3, 4,
       0, 4, 1}
       
       //1, 2, 3,
       //1, 3, 4},
  };

  pyramid.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  pyramid.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  initial = pyramid;
  target  = pyramid;
  
  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.5f,-5.0f));
  LiteMath::float4x4 mRotate1   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-40.0f);
  LiteMath::float4x4 mRotate2   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*+30.0f);
  
  auto mTransform1 = mTranslate*mRotate1;
  auto mTransform2 = mTranslate*mRotate2;

  for(auto& v : initial.vertices)
    v = mTransform1*v;

  for(auto& v : target.vertices)
    v = mTransform2*v;
}