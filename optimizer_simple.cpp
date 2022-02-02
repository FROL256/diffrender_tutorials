#include "optimizer.h"

#include <random>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace LiteMath;

struct OptSimple : public IOptimizer
{
  OptSimple(){}

  void         Init(const TriangleMesh& a_mesh, const Img& a_image) override;
  TriangleMesh Run (size_t a_numIters = 100) override;

protected:
  
  float EvalFunction(const TriangleMesh& mesh, DTriangleMesh& gradMesh);

  TriangleMesh g_mesh; ///<! global mesh optimized mesh
  Img          g_targetImage;
  size_t       g_iter = 0;
};

IOptimizer* CreateSimpleOptimizer() { return new OptSimple; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float OptSimple::EvalFunction(const TriangleMesh& mesh, DTriangleMesh& gradMesh)
{
  constexpr int samples_per_pixel = 4;

  Img img(256, 256);
  std::mt19937 rng(1234);
  render(mesh, samples_per_pixel, rng, img);
  
  std::stringstream strOut;
  strOut  << "rendered_opt/render_" << std::setfill('0') << std::setw(4) << g_iter << ".bmp";
  save_img(img, strOut.str());

  Img adjoint(img.width, img.height, float3{1, 1, 1});
  float mse = MSEAndDiff(img, g_targetImage, adjoint);
  Img dx(img.width, img.height), dy(img.width, img.height); // actually not needed here
  
  gradMesh.clear();
  d_render(mesh, adjoint, samples_per_pixel, img.width * img.height , rng, dx, dy, gradMesh);

  g_iter++;
  return mse;
}

void OptSimple::Init(const TriangleMesh& a_mesh, const Img& a_image) 
{ 
  g_mesh = a_mesh; 
  g_targetImage = a_image; 
  g_iter = 0; 
}

TriangleMesh OptSimple::Run(size_t a_numIters) 
{ 
  const size_t eachPassDescreasStep = a_numIters/10; 

  DTriangleMesh gradMesh(g_mesh.vertices.size(), g_mesh.colors.size());
  //float currError = 1e38f;
  float alphaPos   = 0.1f;
  float alphaColor = 0.00001f;
  for(size_t iter=0; iter < a_numIters; iter++)
  {
    float error = EvalFunction(g_mesh, gradMesh);
    std::cout << "iter " << iter << ", error = " << error << std::endl;
   
    //PrintMesh(gradMesh);
    for(int vertId=0; vertId< g_mesh.vertices.size(); vertId++)
      g_mesh.vertices[vertId] -= gradMesh.vertices()[vertId]*alphaPos;
    for(int faceId=0; faceId < g_mesh.colors.size(); faceId++)
      g_mesh.colors[faceId] -= gradMesh.faceColors()[faceId]*alphaColor;

    if(iter % eachPassDescreasStep == 0)
    {
      alphaPos   = alphaPos*0.75f;
      alphaColor = alphaColor*0.5f;
    }
  }

  return g_mesh;
}