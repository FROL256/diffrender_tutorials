#include "optimizer.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace LiteMath;

struct OptSimple : public IOptimizer
{
  OptSimple(){}

  void         Init(const TriangleMesh& a_mesh, const Img& a_image, OptimizerParameters a_params) override;
  TriangleMesh Run (size_t a_numIters = 100) override;

protected:
  
  float EvalFunction(const TriangleMesh& mesh, DTriangleMesh& gradMesh);
  void  OptStep(const DTriangleMesh &gradMesh, TriangleMesh* mesh, const GammaVec& a_gamma);

  TriangleMesh m_mesh; ///<! global mesh optimized mesh
  Img          m_targetImage;
  size_t       m_iter = 0;
  OptimizerParameters m_params;

  std::vector<GradReal> m_GSquare; ///<! m_GSquare is a vector of the sum of the squared gradients at or before iteration i
};

IOptimizer* CreateSimpleOptimizer() { return new OptSimple; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);

void  OptSimple::OptStep(const DTriangleMesh &gradMesh, TriangleMesh* mesh, const GammaVec& a_gamma)
{
  if(m_params.alg == GD_Naive)
  {
    for(int vertId=0; vertId< mesh->vertices.size(); vertId++)
      mesh->vertices[vertId] -= gradMesh.vert_at(vertId)*a_gamma.pos; //*float3(1,1,1);
    
    for(int faceId=0; faceId < mesh->colors.size(); faceId++)
      mesh->colors[faceId] -= gradMesh.color_at(faceId)*a_gamma.color;
  }
  else if(m_params.alg == GD_AdaGrad)
  {
    //adam_vec_v += BMO_MATOPS_POW(grad_p,2);
    for(size_t i=0;i<gradMesh.size();i++)
      m_GSquare[i] += (gradMesh[i]*gradMesh[i]);
    
    //direc_out = BMO_MATOPS_ARRAY_DIV_ARRAY( gd_settings.par_step_size * grad_p, BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_SQRT(adam_vec_v), gd_settings.par_ada_norm_term) );

  }
}

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

void OptSimple::Init(const TriangleMesh& a_mesh, const Img& a_image, OptimizerParameters a_params) 
{ 
  m_mesh        = a_mesh; 
  m_targetImage = a_image; 
  m_iter        = 0; 
  m_params      = a_params;
}

TriangleMesh OptSimple::Run(size_t a_numIters) 
{ 
  DTriangleMesh gradMesh(m_mesh.vertices.size(), m_mesh.colors.size(), m_mesh.m_meshType, m_mesh.m_geomType);
  
  if(int(m_params.alg) >= int(GD_AdaGrad)) {
    m_GSquare.resize(gradMesh.size());
    memset(m_GSquare.data(), 0, sizeof(GradReal)*m_GSquare.size());
  }

  auto gamma = gradMesh.getGamma(m_targetImage.width());

  for(size_t iter=0; iter < a_numIters; iter++)
  {
    float error = EvalFunction(m_mesh, gradMesh);
    std::cout << "iter " << iter << ", error = " << error << std::endl;
   
    //PrintMesh(gradMesh);
    OptStep(gradMesh, &m_mesh, gamma);
    
    if(iter > 50 && iter % m_params.decayPeriod == 0) {
      gamma.pos   = gamma.pos*0.5f;
      gamma.color = gamma.color*0.75f;
    }
    else if(iter % m_params.decayPeriod == 0) {
      gamma.pos   = gamma.pos*0.75f;
      gamma.color = gamma.color*0.75f;
    }
  }

  return m_mesh;
}