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

using std::for_each;
using std::upper_bound;
using std::vector;
using std::string;
using std::min;
using std::max;
using std::set;
using std::fstream;

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::int2;

using LiteMath::clamp;
using LiteMath::normalize;

constexpr static int  SAM_PER_PIXEL = 16;

unsigned int g_table[qmc::QRNG_DIMENSIONS][qmc::QRNG_RESOLUTION];
float g_hammSamples[2*SAM_PER_PIXEL];

std::shared_ptr<IRayTracer> g_tracer = nullptr;
CamInfo g_uniforms;

void glhFrustumf3(float *matrix, float left, float right, float bottom, float top, float znear, float zfar)
{
  float temp, temp2, temp3, temp4;
  temp = 2.0f * znear;
  temp2 = right - left;
  temp3 = top - bottom;
  temp4 = zfar - znear;
  matrix[0] = temp / temp2;
  matrix[1] = 0.0;
  matrix[2] = 0.0;
  matrix[3] = 0.0;
  matrix[4] = 0.0;
  matrix[5] = temp / temp3;
  matrix[6] = 0.0;
  matrix[7] = 0.0;
  matrix[8] = (right + left) / temp2;
  matrix[9] = (top + bottom) / temp3;
  matrix[10] = (-zfar - znear) / temp4;
  matrix[11] = -1.0;
  matrix[12] = 0.0;
  matrix[13] = 0.0;
  matrix[14] = (-temp * zfar) / temp4;
  matrix[15] = 0.0;
}

// matrix will receive the calculated perspective matrix. You would have to upload to your shader or use glLoadMatrixf if you aren't using shaders
//
void glhPerspectivef3(float *matrix, float fovy, float aspectRatio, float znear, float zfar)
{
  const float ymax = znear * std::tan(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspectRatio;
  glhFrustumf3(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
}

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

  qmc::init(g_table);
  qmc::planeHammersley(g_hammSamples, SAM_PER_PIXEL);

  Img img(256, 256);

  g_uniforms.width  = float(img.width());
  g_uniforms.height = float(img.height());
  glhPerspectivef3(g_uniforms.projM, 45.0f, g_uniforms.width / g_uniforms.height, 0.1f, 100.0f);

  TriangleMesh initialMesh, targetMesh;
  //scn01_TwoTrisFlat(initialMesh, targetMesh);
  //scn02_TwoTrisSmooth(initialMesh, targetMesh);
  //scn03_Triangle3D_White(initialMesh, targetMesh);
  //scn04_Triangle3D_Colored(initialMesh, targetMesh); // bad
  //scn05_Pyramid3D(initialMesh, targetMesh);
  //scn06_Cube3D_VColor(initialMesh, targetMesh);      // bad
  scn07_Cube3D_FColor(initialMesh, targetMesh);      
  
  if(initialMesh.m_geomType == GEOM_TYPES::TRIANGLE_2D)
    g_tracer = MakeRayTracer2D("");  
  else
    g_tracer = MakeRayTracer3D("");

  auto pDRender = MakeDifferentialRenderer(initialMesh, SAM_PER_PIXEL);

  if(1)
  {
    Img initial(img.width(), img.height(), float3{0, 0, 0});
    Img target(img.width(), img.height(), float3{0, 0, 0});
    //render(initialMesh, SAM_PER_PIXEL, initial);
    //render(targetMesh, SAM_PER_PIXEL, target);
    pDRender->prepare(initialMesh);
    pDRender->render(initialMesh, g_uniforms, initial);

    pDRender->prepare(targetMesh);
    pDRender->render(targetMesh, g_uniforms, target);
    
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
    pDRender->prepare(targetMesh);
    pDRender->render(targetMesh, g_uniforms, target);
    
    pDRender->prepare(initialMesh);
    pDRender->render(initialMesh, g_uniforms, img);
    
    DTriangleMesh grad1(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);
    DTriangleMesh grad2(initialMesh.vertices.size(), initialMesh.indices.size()/3, initialMesh.m_meshType, initialMesh.m_geomType);

    LossAndDiffLoss(img, target, adjoint); // put MSE ==> adjoint 
    pDRender->d_render(initialMesh, g_uniforms, adjoint, img.width()*img.height(), 
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
    return 0;
  }

  img.clear(float3{0,0,0});
  pDRender->prepare(targetMesh);
  pDRender->render(targetMesh, g_uniforms, img);
  LiteImage::SaveImage("rendered_opt/z_target.bmp", img);
  
  #ifdef COMPLEX_OPT
  IOptimizer* pOpt = CreateComplexOptimizer();
  #else
  IOptimizer* pOpt = CreateSimpleOptimizer();
  #endif

  //pOpt->Init(initialMesh, img, {30,GD_Naive}); 
  pOpt->Init(initialMesh, img, pDRender, g_uniforms, {100,GD_Adam}); 

  TriangleMesh mesh3 = pOpt->Run(300);
  
  img.clear(float3{0,0,0});
  pDRender->prepare(mesh3);
  pDRender->render(mesh3, g_uniforms, img);
  LiteImage::SaveImage("rendered_opt/z_target2.bmp", img);
  
  delete pOpt; pOpt = nullptr;
  return 0;
}
