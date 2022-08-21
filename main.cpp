#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include "LiteMath.h"

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

#include <cassert>
#include <iomanip>

#include "dmesh.h"
#include "functions.h"
#include "raytrace.h"

#include "optimizer.h"
#include "scenes.h"

#include "qmc.h"
#include "drender.h"

using std::vector;
using std::string;
using std::min;
using std::max;
using std::set;
using std::fstream;

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;

using LiteMath::clamp;
using LiteMath::normalize;

constexpr static int  SAM_PER_PIXEL = 16;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PrintMesh(const DTriangleMesh& a_mesh)
{
  for(int i=0; i<a_mesh.numVerts();i++)
    std::cout << "ver[" << i << "]: " << a_mesh.vert_at(i).x << ", " << a_mesh.vert_at(i).y << std::endl;  
  std::cout << std::endl;
  for(size_t i=0; i<a_mesh.numFaces();i++)
    std::cout << "col[" << i << "]: " << a_mesh.color_at(i).x << ", " << a_mesh.color_at(i).y << ", " << a_mesh.color_at(i).z << std::endl;
  std::cout << std::endl;
}

void PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2);

float LossAndDiffLoss(const Img& b, const Img& a, Img& a_outDiff)
{
  assert(a.width()*a.height() == b.width()*b.height());
  double accumMSE = 0.0f;
  const size_t imgSize = a.width()*a.height();
  for(size_t i=0;i<imgSize;i++)
  {
    const float3 diffVec = b.data()[i] - a.data()[i];
    a_outDiff.data()[i] = 2.0f*diffVec;                    // (I[x,y] - I_target[x,y])    // dirrerential of the loss function 
    accumMSE += double(dot(diffVec, diffVec));             // (I[x,y] - I_target[x,y])^2  // the loss function itself
  }
  return float(accumMSE);
}

float MSE(const Img& b, const Img& a) { return LiteImage::MSE(b,a); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);


void d_finDiff2(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IOptimizer* CreateSimpleOptimizer(); 
IOptimizer* CreateComplexOptimizer();

int main(int argc, char *argv[]) 
{
  #ifdef WIN32
  mkdir("rendered");
  mkdir("rendered_opt");
  mkdir("fin_diff");
  #else
  mkdir("rendered",     S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir("rendered_opt", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  mkdir("fin_diff",     S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  #endif

  Img img(256, 256);
  
  constexpr int camsNum = 3;
  CamInfo cameras[camsNum] = {};
  for(int i=0;i<camsNum;i++) {
    cameras[i].width  = float(img.width());
    cameras[i].height = float(img.height());
    cameras[i].mWorldView.identity();
    cameras[i].mProj.identity();
  }

  float4x4 mProj = LiteMath::perspectiveMatrix(45.0f, cameras[0].width / cameras[0].height, 0.1f, 100.0f);

  cameras[0].mProj      = mProj;
  cameras[0].mWorldView = LiteMath::translate4x4(float3(0,0,-5));

  cameras[1].mProj      = mProj;
  cameras[1].mWorldView = LiteMath::translate4x4(float3(0,0,-5))*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*120.0f)*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*45.0f);

  cameras[2].mProj      = mProj;
  cameras[2].mWorldView = LiteMath::translate4x4(float3(0,0,-5))*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*(-120.0f))*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*(-45.0f));

  for(int i=0;i<camsNum;i++)
    cameras[i].commit();

  auto g_uniforms = cameras[0];

  TriangleMesh initialMesh, targetMesh;
  //scn01_TwoTrisFlat(initialMesh, targetMesh);
  //scn02_TwoTrisSmooth(initialMesh, targetMesh);
  //scn03_Triangle3D_White(initialMesh, targetMesh);
  //scn04_Triangle3D_Colored(initialMesh, targetMesh); // bad
  //scn05_Pyramid3D(initialMesh, targetMesh);
  //scn06_Cube3D_VColor(initialMesh, targetMesh);      // bad
  scn07_Cube3D_FColor(initialMesh, targetMesh);      

  auto pDRender = MakeDifferentialRenderer(initialMesh, SAM_PER_PIXEL);

  if(0)
  {
    Img initial(img.width(), img.height(), float3{0, 0, 0});
    Img target(img.width(), img.height(), float3{0, 0, 0});
    //render(initialMesh, SAM_PER_PIXEL, initial);
    //render(targetMesh, SAM_PER_PIXEL, target);
    pDRender->commit(initialMesh);
    pDRender->render(initialMesh, cameras, &initial, 1);

    pDRender->commit(targetMesh);
    pDRender->render(targetMesh, cameras, &target, 1);
    
    LiteImage::SaveImage("rendered/initial.bmp", initial);
    LiteImage::SaveImage("rendered/target.bmp", target);
    //return 0;
  }

  if(0) // check gradients with finite difference method
  {
    Img target(img.width(), img.height(), float3{0, 0, 0});
    Img adjoint(img.width(), img.height(), float3{0, 0, 0});
    
    Img dxyzDebug[3];
    for(int i=0;i<3;i++)
      dxyzDebug[i] = Img(img.width(), img.height(), float3{0, 0, 0});

    //render(initialMesh, SAM_PER_PIXEL, img);
    //render(targetMesh, SAM_PER_PIXEL, target);
    pDRender->commit(targetMesh);
    pDRender->render(targetMesh, cameras, &target, 1);
    
    pDRender->commit(initialMesh);
    pDRender->render(initialMesh, cameras, &img, 1);
    
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    LossAndDiffLoss(img, target, adjoint); // put MSE ==> adjoint 
    pDRender->d_render(initialMesh, cameras, &adjoint, 1, img.width()*img.height(), 
                       grad1, dxyzDebug, 3);

    for(int i=0;i<3;i++)
    {
      std::stringstream strOut;
      strOut << "our_diff/pos_xyz_" << G_DEBUG_VERT_ID+i << ".bmp";
      auto path = strOut.str();
      LiteImage::SaveImage(path.c_str(), dxyzDebug[i]);
    }

    const float dPos = (initialMesh.m_geomType == GEOM_TYPES::TRIANGLE_2D) ? 1.0f : 2.0f/float(img.width());
    //d_finDiff (initialMesh, "fin_diff", img, target,  pDRender, g_uniforms, grad2, dPos, 0.01f);
    d_finDiff2(initialMesh, "fin_diff", img, target, pDRender, g_uniforms, grad2, dPos, 0.01f);
    
    PrintAndCompareGradients(grad1, grad2);
    return 0;
  }

  if(1) // check gradients for different image views
  {
    std::vector<Img> targets(camsNum), images(camsNum), adjoints(camsNum);
    for(int i=0;i<camsNum;i++) {
      targets [i] = Img(img.width(), img.height(), float3{0, 0, 0});
      images  [i] = Img(img.width(), img.height(), float3{0, 0, 0});
      adjoints[i] = Img(img.width(), img.height(), float3{0, 0, 0});
    }
  
    pDRender->commit(targetMesh);
    pDRender->render(targetMesh, cameras, targets.data(), camsNum);
    
    for(int i=0;i<camsNum;i++) {
      std::stringstream strOut;
      strOut << "rendered/target" << i << ".bmp";
      auto fileName = strOut.str();
      LiteImage::SaveImage(fileName.c_str(), targets[i]);
    }

    pDRender->commit(initialMesh);
    pDRender->render(initialMesh, cameras, images.data(), camsNum);

    for(int i=0;i<camsNum;i++) {
      std::stringstream strOut;
      strOut << "rendered/initial" << i << ".bmp";
      auto fileName = strOut.str();
      LiteImage::SaveImage(fileName.c_str(), images[i]);
    }

    for(int i=0;i<camsNum;i++)
      LossAndDiffLoss(images[i], targets[i], adjoints[i]); 
  
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    pDRender->d_render(initialMesh, cameras+0, &adjoints[0], 1, img.width()*img.height(), grad1);
    pDRender->d_render(initialMesh, cameras+1, &adjoints[1], 1, img.width()*img.height(), grad2);
    
    DTriangleMesh grad_avg(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    for(size_t i=0;i<grad_avg.size();i++)
      grad_avg[i] = 0.5f*(grad1[i] + grad2[i]);
    
    DTriangleMesh grad12(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    pDRender->d_render(initialMesh, cameras+0, &adjoints[0], 2, img.width()*img.height(), grad12);
    
    PrintAndCompareGradients(grad1, grad2);
    std::cout << "********************************************" << std::endl;
    std::cout << "********************************************" << std::endl;
    PrintAndCompareGradients(grad12, grad_avg);
    return 0;
  }

  
  Img targets[camsNum];
  for(int i=0;i<camsNum;i++) {
    targets[i].resize(img.width(), img.height());
    targets[i].clear(float3{0,0,0});
  }

  pDRender->commit(targetMesh);
  pDRender->render(targetMesh, cameras, targets, camsNum);
  LiteImage::SaveImage("rendered_opt/z_target.bmp", targets[0]);
  
  #ifdef COMPLEX_OPT
  IOptimizer* pOpt = CreateComplexOptimizer();
  #else
  IOptimizer* pOpt = CreateSimpleOptimizer();
  #endif

  //pOpt->Init(initialMesh, img, {30,GD_Naive}); 
  pOpt->Init(initialMesh, pDRender, cameras, targets, 1, {100,GD_Adam}); //  {30,GD_Naive}

  TriangleMesh mesh3 = pOpt->Run(300);
  
  img.clear(float3{0,0,0});
  pDRender->commit(mesh3);
  pDRender->render(mesh3, cameras, &img, 1);
  LiteImage::SaveImage("rendered_opt/z_target2.bmp", img);
  
  delete pOpt; pOpt = nullptr;
  return 0;
}
