#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include "LiteMath.h"
using namespace LiteMath;

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

#include <cassert>
#include <iomanip>

#include "dmesh.h"

constexpr static int SAM_PER_PIXEL = 16;

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
                 DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f) 
{
  Img img(origin.width(), origin.height());

  d_mesh.resize(mesh.vertices.size(), mesh.indices.size()/3, mesh.type);
  d_mesh.clear();
  
  const float MSEOrigin = MSE(origin, target);
  const float scale = float(256*256*3);

  for(size_t i=0; i<mesh.vertices.size();i++)
  {
    TriangleMesh copy;
    
    // dx
    //
    copy = mesh;
    copy.vertices[i].x += dPos;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    auto diffToTarget = (MSE(img,target) - MSEOrigin)/dPos;
    d_mesh.vertices_s()[i*2+0] += GradReal(diffToTarget*scale);
    
    // dy
    //
    copy = mesh;
    copy.vertices[i].y += dPos;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);

    diffToTarget = (MSE(img,target) - MSEOrigin)/dPos;
    d_mesh.vertices_s()[2*i+1] += GradReal(diffToTarget*scale);
  }
  
  size_t colrsNum = (mesh.type == TRIANGLE_2D_VERT_COL) ? mesh.vertices.size() : mesh.indices.size()/3;
  
  for(size_t i=0; i<colrsNum;i++)
  {
    TriangleMesh copy;
    
    // d_red
    //
    copy = mesh;
    copy.colors[i].x += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    auto diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+0] += GradReal(diffToTarget*scale);

    // d_green
    //
    copy = mesh;
    copy.colors[i].y += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+1] += GradReal(diffToTarget*scale);

    // d_blue
    //
    copy = mesh;
    copy.colors[i].z += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+2] += GradReal(diffToTarget*scale);
  }

}

void d_finDiff2(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
                DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f) 
{
  Img img(origin.width(), origin.height());

  d_mesh.resize(mesh.vertices.size(), mesh.indices.size()/3, mesh.type);
  d_mesh.clear();
  
  const Img MSEOrigin = LiteImage::MSEImage(origin, target);

  for(size_t i=0; i<mesh.vertices.size();i++)
  {
    TriangleMesh copy;
    
    // dx
    //
    copy = mesh;
    copy.vertices[i].x += dPos;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    auto diffImage = (LiteImage::MSEImage(img,target) - MSEOrigin)/dPos;   
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "posx_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffImage);
    }
    float3 summColor = SummOfPixels(diffImage); 
    d_mesh.vertices_s()[i*2+0] += GradReal(summColor.x + summColor.y + summColor.z);
    
    // dy
    //
    copy = mesh;
    copy.vertices[i].y += dPos;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);

    diffImage = (LiteImage::MSEImage(img,target) - MSEOrigin)/dPos;   
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "posy_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffImage);
    }
    summColor = SummOfPixels(diffImage); 
    d_mesh.vertices_s()[i*2+1] += GradReal(summColor.x + summColor.y + summColor.z);
  }
  
  size_t colrsNum = (mesh.type == TRIANGLE_2D_VERT_COL) ? mesh.vertices.size() : mesh.indices.size()/3;
  
  for(size_t i=0; i<colrsNum;i++)
  {
    TriangleMesh copy;
    
    // d_red
    //
    copy = mesh;
    copy.colors[i].x += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    auto diffToTarget = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "colr_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffToTarget);
    }
    float3 summColor = SummOfPixels(diffToTarget); 
    d_mesh.colors_s()[i*3+0] += GradReal(summColor.x + summColor.y + summColor.z);

    // d_green
    //
    copy = mesh;
    copy.colors[i].y += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    diffToTarget = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "colg_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffToTarget);
    }
    summColor = SummOfPixels(diffToTarget); 
    d_mesh.colors_s()[i*3+1] += GradReal(summColor.x + summColor.y + summColor.z);

    // d_blue
    //
    copy = mesh;
    copy.colors[i].z += dCol;
    img.clear(float3{0,0,0});
    render(copy, SAM_PER_PIXEL, img);
    
    diffToTarget = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "colb_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffToTarget); // 
    }
    summColor = SummOfPixels(diffToTarget); 
    d_mesh.colors_s()[i*3+2] += GradReal(summColor.x + summColor.y + summColor.z);
  }

}