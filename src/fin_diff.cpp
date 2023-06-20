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
#include "drender.h"

constexpr static int SAM_PER_PIXEL = 16;

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f) 
{
  Img img(origin.width(), origin.height());

  d_mesh.reset(mesh);
  
  const float MSEOrigin = MSE(origin, target);
  const float scale = float(256*256*3);

  for(size_t i=0; i<mesh.vertex_count();i++)
  {
    TriangleMesh copy;
    
    // dx
    //
    copy = mesh;
    copy.vertices[i].x += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffToTarget = (MSE(img,target) - MSEOrigin)/dPos;
    d_mesh.vertices_s()[i*3+0] += GradReal(diffToTarget*scale);
    
    // dy
    //
    copy = mesh;
    copy.vertices[i].y += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);

    diffToTarget = (MSE(img,target) - MSEOrigin)/dPos;
    d_mesh.vertices_s()[3*i+1] += GradReal(diffToTarget*scale);

    // dz 
    //
    copy = mesh;
    copy.vertices[i].z += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    diffToTarget = (MSE(img,target) - MSEOrigin)/dPos;
    d_mesh.vertices_s()[3*i+2] += GradReal(diffToTarget*scale);
  }
  
  size_t colrsNum = mesh.vertex_count();
  
  for(size_t i=0; i<colrsNum;i++)
  {
    TriangleMesh copy;
    
    // d_red
    //
    copy = mesh;
    copy.colors[i].x += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+0] += GradReal(diffToTarget*scale);

    // d_green
    //
    copy = mesh;
    copy.colors[i].y += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+1] += GradReal(diffToTarget*scale);

    // d_blue
    //
    copy = mesh;
    copy.colors[i].z += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    diffToTarget = (MSE(img,target) - MSEOrigin)/dCol;
    d_mesh.colors_s()[i*3+2] += GradReal(diffToTarget*scale);
  }

}

void d_finDiff2(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
                DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f) 
{
  Img img(origin.width(), origin.height());

  d_mesh.reset(mesh);
  
  const Img MSEOrigin = LiteImage::MSEImage(origin, target);

  for(size_t i=0; i<mesh.vertex_count();i++)
  {
    TriangleMesh copy;
    
    // dx
    //
    copy = mesh;
    copy.vertices[i].x += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffImageX = (LiteImage::MSEImage(img,target) - MSEOrigin)/dPos;   
    float3 summColor = SummOfPixels(diffImageX); 
    d_mesh.vertices_s()[i*3+0] += GradReal(summColor.x + summColor.y + summColor.z);
    
    // dy
    //
    copy = mesh;
    copy.vertices[i].y += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);

    auto diffImageY = (LiteImage::MSEImage(img,target) - MSEOrigin)/dPos;   
    summColor = SummOfPixels(diffImageY); 
    d_mesh.vertices_s()[i*3+1] += GradReal(summColor.x + summColor.y + summColor.z);

    // dz 
    //
    copy = mesh;
    copy.vertices[i].z += dPos;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);

    auto diffImageZ = (LiteImage::MSEImage(img,target) - MSEOrigin)/dPos;   
    Img diffImage(diffImageX.width(), diffImageX.height()); 
    for(int y=0;y<diffImageX.height();y++)
      for(int x=0;x<diffImageX.width();x++)
        diffImage[int2(x,y)] = float3(diffImageX[int2(x,y)].x, diffImageY[int2(x,y)].x, diffImageZ[int2(x,y)].x);

    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "pos_xyz_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffImage);
    }
    summColor = SummOfPixels(diffImageZ); 
    d_mesh.vertices_s()[i*3+2] += GradReal(summColor.x + summColor.y + summColor.z);
  }
  
  size_t colrsNum = mesh.vertex_count();
  
  for(size_t i=0; i<colrsNum;i++)
  {
    TriangleMesh copy;
    
    // d_red
    //
    copy = mesh;
    copy.colors[i].x += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffToTargetX = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    float3 summColor = SummOfPixels(diffToTargetX); 
    d_mesh.colors_s()[i*3+0] += GradReal(summColor.x + summColor.y + summColor.z);

    // d_green
    //
    copy = mesh;
    copy.colors[i].y += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffToTargetY = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    summColor = SummOfPixels(diffToTargetY); 
    d_mesh.colors_s()[i*3+1] += GradReal(summColor.x + summColor.y + summColor.z);

    // d_blue
    //
    copy = mesh;
    copy.colors[i].z += dCol;
    img.clear(float3{0,0,0});
    a_pDRImpl->commit(copy);
    a_pDRImpl->render(copy, &a_camData, &img, 1);
    
    auto diffToTargetZ = (LiteImage::MSEImage(img,target) - MSEOrigin)/dCol;
    Img diffImage(diffToTargetX.width(), diffToTargetX.height()); 
    for(int y=0;y<diffToTargetX.height();y++)
      for(int x=0;x<diffToTargetX.width();x++)
        diffImage[int2(x,y)] = float3(diffToTargetX[int2(x,y)].x + diffToTargetX[int2(x,y)].y + diffToTargetX[int2(x,y)].z, 
                                      diffToTargetY[int2(x,y)].x + diffToTargetY[int2(x,y)].y + diffToTargetY[int2(x,y)].z, 
                                      diffToTargetZ[int2(x,y)].x + diffToTargetZ[int2(x,y)].y + diffToTargetZ[int2(x,y)].z)*0.3334f;

    if(outFolder != nullptr)
    {
      std::stringstream strOut;
      strOut << outFolder << "/" << "col_" << i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), diffImage); // 
    }
    summColor = SummOfPixels(diffToTargetZ); 
    d_mesh.colors_s()[i*3+2] += GradReal(summColor.x + summColor.y + summColor.z);
  }

}

void PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2)
{
  double totalError = 0.0;
  double posError = 0.0;
  double colError = 0.0;
  double posLengthL1 = 0.0;
  double colLengthL1 = 0.0;

  auto subvecPos1 = grad1.subvecPos();
  auto subvecCol1 = grad1.subvecCol();

  auto subvecPos2 = grad2.subvecPos();
  auto subvecCol2 = grad2.subvecCol();

  for(size_t i=0;i<subvecPos1.size();i++) {
    double diff = std::abs(double(subvecPos1[i] - subvecPos2[i]));
    posError    += diff;
    totalError  += diff;
    posLengthL1 += std::abs(subvecPos2[i]);
    std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[i] << "\t";  
    std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[i] << std::endl;
  }

  std::cout << "--------------------------" << std::endl;
  for(size_t i=0;i<subvecCol1.size();i++) {
    double diff = std::abs(double(subvecCol1[i] - subvecCol2[i]));
    colError   += diff;
    totalError += diff;
    colLengthL1 += std::abs(subvecCol2[i]);
    std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[subvecPos1.size() + i] << "\t";  
    std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[subvecPos1.size() + i] << std::endl;
  }
  
  std::cout << "==========================" << std::endl;
  std::cout << "GradErr[L1](vpos ) = " << posError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(posError/posLengthL1) << "%" << std::endl;
  std::cout << "GradErr[L1](color) = " << colError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(colError/colLengthL1) << "%" << std::endl;
  std::cout << "GradErr[L1](total) = " << totalError/double(grad1.size()) << "\t which is \t" << 100.0*(totalError/(posLengthL1+colLengthL1)) << "%" << std::endl;
}