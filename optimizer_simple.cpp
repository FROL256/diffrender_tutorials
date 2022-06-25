#include "optimizer.h"

#include <random>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void opt_step(const DTriangleMesh &gradMesh, float alphaPos, float alphaColor, 
              TriangleMesh* mesh)
{
  for(int vertId=0; vertId< mesh->vertices.size(); vertId++)
    mesh->vertices[vertId] -= gradMesh.vertices()[vertId]*alphaPos;
  
  for(int faceId=0; faceId < mesh->colors.size(); faceId++)
    mesh->colors[faceId] -= gradMesh.colors()[faceId]*alphaColor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace LiteMath;

struct OptSimple : public IOptimizer
{
  OptSimple(){}

  void         Init(const TriangleMesh& a_mesh, const Img& a_image) override;
  TriangleMesh Run (size_t a_numIters = 100) override;

protected:
  
  float EvalFunction(const TriangleMesh& mesh, DTriangleMesh& gradMesh);

  TriangleMesh m_mesh; ///<! global mesh optimized mesh
  Img          m_targetImage;
  size_t       m_iter = 0;
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
  strOut  << "rendered_opt/render_" << std::setfill('0') << std::setw(4) << m_iter << ".bmp";
  auto temp = strOut.str();
  save_img(img, temp.c_str());

  Img adjoint(img.width(), img.height(), float3{1, 1, 1});
  float mse = MSEAndDiff(img, m_targetImage, adjoint);
  
  gradMesh.clear();
  d_render(mesh, adjoint, samples_per_pixel, img.width() * img.height(), rng, nullptr, nullptr, 
           gradMesh);

  m_iter++;
  return mse;
}

void OptSimple::Init(const TriangleMesh& a_mesh, const Img& a_image) 
{ 
  m_mesh = a_mesh; 
  m_targetImage = a_image; 
  m_iter = 0; 
}

TriangleMesh OptSimple::Run(size_t a_numIters) 
{ 
  const size_t eachPassDescreasStep = a_numIters/10; 

  DTriangleMesh gradMesh(m_mesh.vertices.size(), m_mesh.colors.size());
  //float currError = 1e38f;
  float alphaPos   = 0.1f;
  float alphaColor = 0.0001f;
  for(size_t iter=0; iter < a_numIters; iter++)
  {
    float error = EvalFunction(m_mesh, gradMesh);
    std::cout << "iter " << iter << ", error = " << error << std::endl;
   
    //PrintMesh(gradMesh);
    opt_step(gradMesh, alphaPos, alphaColor, 
             &m_mesh);
    
    if(iter > 100 && iter % eachPassDescreasStep == 0)
    {
      alphaPos   = alphaPos*0.5f;
      alphaColor = alphaColor*0.5f;
    }
    else if(iter % eachPassDescreasStep == 0)
    {
      alphaPos   = alphaPos*0.75f;
      alphaColor = alphaColor*0.75f;
    }
  }

  return m_mesh;
}