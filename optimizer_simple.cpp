#include "optimizer.h"

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
    mesh->vertices[vertId] -= gradMesh.vert_at(vertId)*alphaPos; //*float3(1,1,1);
  
  for(int faceId=0; faceId < mesh->colors.size(); faceId++)
    mesh->colors[faceId] -= gradMesh.color_at(faceId)*alphaColor;
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

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);

float OptSimple::EvalFunction(const TriangleMesh& mesh, DTriangleMesh& gradMesh)
{
  const int samples_per_pixel = 16;

  Img img(256, 256);
  render(mesh, samples_per_pixel, img);
  
  std::stringstream strOut;
  strOut  << "rendered_opt/render_" << std::setfill('0') << std::setw(4) << m_iter << ".bmp";
  auto temp = strOut.str();
  LiteImage::SaveImage(temp.c_str(), img);

  Img adjoint(img.width(), img.height(), float3{0, 0, 0});
  float mse = LossAndDiffLoss(img, m_targetImage, adjoint);
  
  gradMesh.clear();
  d_render(mesh, adjoint, samples_per_pixel, img.width() * img.height(), nullptr, nullptr, 
           gradMesh);

  //const float dPos = (mesh.m_geomType == TRIANGLE_2D) ? 1.0f : 4.0f/float(img.width());
  //d_finDiff (mesh, "fin_diff", img, m_targetImage, gradMesh, dPos, 0.01f);

  m_iter++;
  return mse;
}

void OptSimple::Init(const TriangleMesh& a_mesh, const Img& a_image) 
{ 
  m_mesh        = a_mesh; 
  m_targetImage = a_image; 
  m_iter        = 0; 
}

TriangleMesh OptSimple::Run(size_t a_numIters) 
{ 
  const size_t eachPassDescreasStep = 30; //a_numIters/10; 

  DTriangleMesh gradMesh(m_mesh.vertices.size(), m_mesh.colors.size());
  float alphaPos   = 0.2f;
  float alphaColor = 4.0f/float(m_targetImage.width()*m_targetImage.height()); 
  //float alphaPos   = 0.01f/float(m_targetImage.width());
  //float alphaColor = 0.0f; 
  for(size_t iter=0; iter < a_numIters; iter++)
  {
    float error = EvalFunction(m_mesh, gradMesh);
    std::cout << "iter " << iter << ", error = " << error << std::endl;
   
    //PrintMesh(gradMesh);
    opt_step(gradMesh, alphaPos, alphaColor, 
             &m_mesh);
    
    if(iter > 50 && iter % eachPassDescreasStep == 0)
    {
      alphaPos   = alphaPos*0.5f;
      alphaColor = alphaColor*0.75f;
    }
    else if(iter % eachPassDescreasStep == 0)
    {
      alphaPos   = alphaPos*0.75f;
      alphaColor = alphaColor*0.75f;
    }
  }

  return m_mesh;
}