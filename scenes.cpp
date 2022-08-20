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

void scn03_Triangle3D_White(TriangleMesh& initial, TriangleMesh& target)
{
   TriangleMesh tridata{
      // vertices
      {{0.0f, 1.0f, 0.0f},    
       {1.0f, -1.0f, 1.0f},
       {-1.0f, -1.0f, 1.0f},  
       },

      // color
      
      //{{0.07805659f, 0.07805659f, 0.07805659f}, 
      // {0.07805659f, 0.07805659f, 0.07805659f}, 
      // {0.07805659f, 0.07805659f, 0.07805659f},
      // },
      
      {{1.0f, 1.0f, 1.0f}, 
       {1.0f, 1.0f, 1.0f}, 
       {1.0f, 1.0f, 1.0f},
       },

      // indices
      {0, 1, 2}
  };

  tridata.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  tridata.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  initial = tridata;
  target  = tridata;
  
  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.5f,0.0f));
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

void scn04_Triangle3D_Colored(TriangleMesh& initial, TriangleMesh& target)
{
   TriangleMesh tridata{
      // vertices
      {{0.0f, 1.0f, 0.0f},    
       {1.0f, -1.0f, 1.0f},
       {-1.0f, -1.0f, 1.0f},  
       },

      //// color
      {{1.0f, 0.0f, 0.0f}, 
       {1.0f, 1.0f, 0.0f}, 
       {0.0f, 0.0f, 1.0f},
       },

      // color
      //{{1.0f, 0.0f, 0.0f}, 
      // {1.0f, 0.0f, 0.0f}, 
      // {0.0f, 0.0f, 0.0f},
      // },

      // indices
      {0, 1, 2}
  };

  TriangleMesh tridata2{
      // vertices
      {{0.0f, 1.0f, 0.0f},    
       {1.0f, -1.0f, 1.0f},
       {-1.0f, -1.0f, 1.0f},  
       },

      // color
      {{0.1f, 1.0f, 0.1f}, 
       {1.0f, 0.1f, 1.0f}, 
       {0.25f, 0.25f, 0.25f},
       },
       

      // indices
      {0, 1, 2}
  };

  tridata.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  tridata.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  tridata2.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  tridata2.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  initial = tridata;
  target  = tridata2;
  
  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.5f,0.0f));
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

void scn05_Pyramid3D(TriangleMesh& initial, TriangleMesh& target)
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
       {0.0f, 1.0f, 1.0f}, 
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
  
  for(auto& c : initial.colors)
    c = float3(0.25f,0.25f,0.25f);

  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.5f,0.0f));
  LiteMath::float4x4 mRotate1   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-40.0f)*LiteMath::rotate4x4Z(LiteMath::DEG_TO_RAD*-30.0f);
  LiteMath::float4x4 mRotate2   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*+50.0f);
  
  auto mTransform1 = mTranslate*mRotate1;
  auto mTransform2 = mTranslate*mRotate2;

  for(auto& v : initial.vertices)
    v = mTransform1*v;

  for(auto& v : target.vertices)
    v = mTransform2*v;
}


void scn06_Cube3D_VColor(TriangleMesh& initial, TriangleMesh& target)
{
  TriangleMesh cube{
    // vertices
    {{1.0f, 1.0f, -1.0f},    // Top
     {-1.0f, 1.0f, -1.0f},  
     {-1.0f, 1.0f, 1.0f},
     {1.0f, 1.0f, 1.0f},
      
     {1.0f, -1.0f, 1.0f},    // Bottom
     {-1.0f, -1.0f, 1.0f},
     {-1.0f, -1.0f, -1.0f},
     {1.0f, -1.0f, -1.0f},

     {1.0f, 1.0f, 1.0f},     // Front
     {-1.0f, 1.0f, 1.0f},
     {-1.0f, -1.0f, 1.0f},
     {1.0f, -1.0f, 1.0f},

     {1.0f, -1.0f, -1.0f},   // Back
     {-1.0f, -1.0f, -1.0f},
     {-1.0f, 1.0f, -1.0f},
     {1.0f, 1.0f, -1.0f},

     {-1.0f, 1.0f, 1.0f},    // Left
     {-1.0f, 1.0f, -1.0f},
     {-1.0f, -1.0f, -1.0f},
     {-1.0f, -1.0f, 1.0f},

     {1.0f, 1.0f, -1.0f},    // Right
     {1.0f, 1.0f, 1.0f}, 
     {1.0f, -1.0f, 1.0f},
     {1.0f, -1.0f, -1.0f},

    },

    // color
    {{0.0f, 1.0f, 0.0f},    // Top
     {0.0f, 1.0f, 0.0f}, 
     {0.0f, 1.0f, 0.0f},
     {0.0f, 1.0f, 0.0f},

     {1.0f, 0.5f, 0.0f},    // Bottom
     {1.0f, 0.5f, 0.0f}, 
     {1.0f, 0.5f, 0.0f},
     {1.0f, 0.5f, 0.0f},

     {1.0f, 0.0f, 0.0f},    // Front
     {1.0f, 0.0f, 0.0f},
     {1.0f, 0.0f, 0.0f},
     {1.0f, 0.0f, 0.0f},

     {1.0f, 1.0f, 0.0f},    // Back
     {1.0f, 1.0f, 0.0f},
     {1.0f, 1.0f, 0.0f},
     {1.0f, 1.0f, 0.0f},

     {0.0f, 0.0f, 1.0f},    // Left
     {0.0f, 0.0f, 1.0f},
     {0.0f, 0.0f, 1.0f},
     {0.0f, 0.0f, 1.0f},

     {1.0f, 0.0f, 1.0f},    // Right
     {1.0f, 0.0f, 1.0f},
     {1.0f, 0.0f, 1.0f},
     {1.0f, 0.0f, 1.0f},

    },

  };

  cube.indices.resize(6*2*3); // 6 faces, 2 triangles per face, 3 indices per triangle 

  for(int face=0; face < 6; face++) // GL_QUADS
  {  
    cube.indices[face*6+0] = face*4+0;
    cube.indices[face*6+1] = face*4+1;
    cube.indices[face*6+2] = face*4+2;

    cube.indices[face*6+3] = face*4+0;
    cube.indices[face*6+4] = face*4+2;
    cube.indices[face*6+5] = face*4+3;
  }

  cube.m_meshType = MESH_TYPES::TRIANGLE_VERT_COL;
  cube.m_geomType = GEOM_TYPES::TRIANGLE_3D;


  initial = cube;
  target  = cube;
  
  for(auto& c : initial.colors)
    c = float3(0.25f, 0.25f, 0.25f);

  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.0f,0.0f));
  LiteMath::float4x4 mRotate1   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-35.0f)*LiteMath::rotate4x4Z(LiteMath::DEG_TO_RAD*-40.0f);
  LiteMath::float4x4 mRotate2   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*60.0f)*LiteMath::rotate4x4Z(LiteMath::DEG_TO_RAD*-20.0f);;
  
  auto mTransform1 = mTranslate*mRotate1;
  auto mTransform2 = mTranslate*mRotate2;

  for(auto& v : initial.vertices)
    v = mTransform1*v;

  for(auto& v : target.vertices)
    v = mTransform2*v;
}


void scn07_Cube3D_FColor(TriangleMesh& initial, TriangleMesh& target)
{
   TriangleMesh cube{
    // vertices
    {{1.0f, 1.0f, -1.0f},    // 0 Top
     {-1.0f, 1.0f, -1.0f},   // 1
     {-1.0f, 1.0f, 1.0f},    // 2
     {1.0f, 1.0f, 1.0f},     // 3
      
     {1.0f, -1.0f, 1.0f},    // 4 Bottom
     {-1.0f, -1.0f, 1.0f},   // 5
     {-1.0f, -1.0f, -1.0f},  // 6
     {1.0f, -1.0f, -1.0f},   // 7
    },

    // color
    {{0.0f, 1.0f, 0.0f},    // Top
     {0.0f, 1.0f, 0.0f}, 
   
     {1.0f, 0.5f, 0.0f},    // Bottom
     {1.0f, 0.5f, 0.0f}, 

     {1.0f, 0.0f, 0.0f},    // Front
     {1.0f, 0.0f, 0.0f},

     {1.0f, 1.0f, 0.0f},    // Back
     {1.0f, 1.0f, 0.0f},

     {0.0f, 0.0f, 1.0f},    // Left
     {0.0f, 0.0f, 1.0f},

     {1.0f, 0.0f, 1.0f},    // Right
     {1.0f, 0.0f, 1.0f},
    },

  };

  cube.indices.resize(6*2*3); // 6 faces, 2 triangles per face, 3 indices per triangle 

  cube.indices[0] = 0; cube.indices[1] = 1; cube.indices[2] = 2;        cube.indices[3] = 0; cube.indices[4] = 2; cube.indices[5] = 3;
  cube.indices[6] = 4; cube.indices[7] = 5; cube.indices[8] = 6;        cube.indices[9] = 4; cube.indices[10] = 6; cube.indices[11] = 7;

  cube.indices[12] = 1; cube.indices[13] = 6; cube.indices[14] = 7;     cube.indices[15] = 1; cube.indices[16] = 7; cube.indices[17] = 0;
  cube.indices[18] = 3; cube.indices[19] = 4; cube.indices[20] = 5;     cube.indices[21] = 3; cube.indices[22] = 5; cube.indices[23] = 2; 

  cube.indices[24] = 2; cube.indices[25] = 1; cube.indices[26] = 6;     cube.indices[27] = 2; cube.indices[28] = 6; cube.indices[29] = 5;
  cube.indices[30] = 0; cube.indices[31] = 3; cube.indices[32] = 4;     cube.indices[33] = 0; cube.indices[34] = 4; cube.indices[35] = 7; 

  cube.m_meshType = MESH_TYPES::TRIANGLE_FACE_COL;
  cube.m_geomType = GEOM_TYPES::TRIANGLE_3D;

  initial = cube;
  target  = cube;
  
  for(auto& c : initial.colors)
    c = float3(0.25f, 0.25f, 0.25f);

  // apply transforms
  //
  LiteMath::float4x4 mTranslate = LiteMath::translate4x4(float3(0,+0.0f,0.0f));
  LiteMath::float4x4 mRotate1   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-35.0f)*LiteMath::rotate4x4Z(LiteMath::DEG_TO_RAD*-15.0f); // TODO: try +15
  LiteMath::float4x4 mRotate2   = LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*-30.0f)*LiteMath::rotate4x4Z(LiteMath::DEG_TO_RAD*-30.0f);
  
  auto mTransform1 = mTranslate*mRotate1;
  auto mTransform2 = mTranslate*mRotate2;

  for(auto& v : initial.vertices)
    v = mTransform1*v;

  for(auto& v : target.vertices)
    v = mTransform2*v;
}