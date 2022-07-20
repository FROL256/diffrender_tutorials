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

  GammaVec EstimateGamma(unsigned imageSize, GEOM_TYPES a_geomType) const;
  void     StepDecay(int a_iterId, GammaVec& a_gamma) const;

  TriangleMesh m_mesh; ///<! global mesh optimized mesh
  Img          m_targetImage;
  size_t       m_iter = 0;
  OptimizerParameters m_params;

  std::vector<GradReal> m_GSquare; ///<! m_GSquare is a vector of the sum of the squared gradients at or before iteration 'i'
  std::vector<GradReal> m_vec; 
};

IOptimizer* CreateSimpleOptimizer() { return new OptSimple; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void d_finDiff(const TriangleMesh &mesh, const char* outFolder, const Img& origin, const Img& target,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);


GammaVec OptSimple::EstimateGamma(unsigned imageSize, GEOM_TYPES a_geomType) const 
{
  GammaVec res(0.2f, 4.0f/float(imageSize*imageSize));

  if(m_params.alg == GD_Naive)
  {
    res = GammaVec(0.2f, 4.0f/float(imageSize*imageSize));
    if(a_geomType == GEOM_TYPES::TRIANGLE_3D)
      res = GammaVec(0.001f/float(imageSize), 0.0);
  }
  else if(m_params.alg >= GD_AdaGrad)
  {
    res = GammaVec(std::sqrt(imageSize)/2, 0.1f);
    if(a_geomType == GEOM_TYPES::TRIANGLE_3D)
      res = GammaVec(0.075f, 0.1f);
  }

  return res;
}

void  OptSimple::StepDecay(int a_iterId, GammaVec& a_gamma) const
{
  if(m_params.alg == GD_Naive)
  {
    if(a_iterId >= 50 && a_iterId % m_params.decayPeriod == 0) {
      a_gamma.pos   = a_gamma.pos*0.5f;
      a_gamma.color = a_gamma.color*0.75f;
    }
    else if(a_iterId % m_params.decayPeriod == 0) {
      a_gamma.pos   = a_gamma.pos*0.75f;
      a_gamma.color = a_gamma.color*0.75f;
    }
  }
  else if(m_params.alg >= GD_AdaGrad)
  {
    if(a_iterId >= 100 && a_iterId % m_params.decayPeriod == 0) {
      a_gamma.pos   = a_gamma.pos*0.75f;
      a_gamma.color = a_gamma.color*0.75f;
    }
  }
}

void  OptSimple::OptStep(const DTriangleMesh &gradMesh, TriangleMesh* mesh, const GammaVec& a_gamma)
{
  if(m_params.alg == GD_Naive)
  { 
    //xNext[i] = x[i] - gamma*gradF[i];
    //
    for(int vertId=0; vertId< mesh->vertices.size(); vertId++)
      mesh->vertices[vertId] -= gradMesh.vert_at(vertId)*a_gamma.pos; //*float3(1,1,1);
    
    for(int faceId=0; faceId < mesh->colors.size(); faceId++)
      mesh->colors[faceId] -= gradMesh.color_at(faceId)*a_gamma.color;
  }
  else if(m_params.alg >= GD_AdaGrad)
  {
    if(m_params.alg == GD_AdaGrad)  // ==> GSquare[i] = gradF[i]*gradF[i]
    {
      for(size_t i=0;i<gradMesh.size();i++)
        m_GSquare[i] += (gradMesh[i]*gradMesh[i]);
    }
    else if(m_params.alg == GD_RMSProp) // ==> GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*gradF[i]*gradF[i]
    {
      const float alpha = 0.5f;
      for(size_t i=0;i<gradMesh.size();i++)
        m_GSquare[i] = 2.0f*(m_GSquare[i]*alpha + (gradMesh[i]*gradMesh[i])*(1.0f-alpha)); // does not works without 2.0f
    }
    else if(m_params.alg == GD_Adam) // ==> Adam(m[i] = b*mPrev[i] + (1-b)*gradF[i], GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*gradF[i]*gradF[i])
    {
      const float alpha = 0.5f;
      const float beta  = 0.25f;
      for(size_t i=0;i<m_vec.size();i++)
        m_vec[i] = m_vec[i]*beta + gradMesh[i]*(1.0f-beta);

      for(size_t i=0;i<gradMesh.size();i++)
        m_GSquare[i] = 2.0f*(m_GSquare[i]*alpha + (gradMesh[i]*gradMesh[i])*(1.0f-alpha)); // does not works without 2.0f

      DTriangleMesh& gradUpdated = const_cast<DTriangleMesh&>(gradMesh);
      for(size_t i=0;i<m_vec.size();i++)
        gradUpdated[i] = m_vec[i];
    }
    
    //xNext[i] = x[i] - gamma/(sqrt(GSquare[i] + epsilon));
    //
    for(int vertId=0; vertId< mesh->vertices.size(); vertId++)
    {
      const GradReal divX = GradReal(1.0)/( std::sqrt(m_GSquare[vertId*3+0] + GradReal(1e-8f)));
      const GradReal divY = GradReal(1.0)/( std::sqrt(m_GSquare[vertId*3+1] + GradReal(1e-8f)));
      const GradReal divZ = GradReal(1.0)/( std::sqrt(m_GSquare[vertId*3+2] + GradReal(1e-8f)));
      mesh->vertices[vertId] -= gradMesh.vert_at(vertId)*float3(divX,divY,divZ)*a_gamma.pos;
    }
    
    const int offset = gradMesh.color_offs();
    for(int faceId=0; faceId < mesh->colors.size(); faceId++)
    {
      const GradReal divX = GradReal(1.0)/( std::sqrt(m_GSquare[offset + faceId*3 + 0] + GradReal(1e-8f)) );
      const GradReal divY = GradReal(1.0)/( std::sqrt(m_GSquare[offset + faceId*3 + 1] + GradReal(1e-8f)) );
      const GradReal divZ = GradReal(1.0)/( std::sqrt(m_GSquare[offset + faceId*3 + 2] + GradReal(1e-8f)) );
      mesh->colors[faceId] -= gradMesh.color_at(faceId)*float3(divX,divY,divZ)*a_gamma.color;
    }
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
  
  if(m_params.alg >= GD_AdaGrad) {
    m_GSquare.resize(gradMesh.size());
    memset(m_GSquare.data(), 0, sizeof(GradReal)*m_GSquare.size());
  }

  if(m_params.alg == GD_Adam) {
    m_vec.resize(gradMesh.size());
    memset(m_vec.data(), 0, sizeof(GradReal)*m_vec.size());
  }

  auto gamma = EstimateGamma(m_targetImage.width(), gradMesh.m_geomType);

  for(size_t iter=0, trueIter = 0; iter < a_numIters; iter++, trueIter++)
  {
    float error = EvalFunction(m_mesh, gradMesh);
    std::cout << "iter " << trueIter << ", error = " << error << std::endl;
    //PrintMesh(gradMesh);
    OptStep(gradMesh, &m_mesh, gamma);
    StepDecay(iter, gamma);

    if(error <= 0.5f && (iter < a_numIters-10)) // perform last 10 iterations and stop
    {
      std::cout << "----------------------------> stop by error, perform last 10 iterations: " << std::endl;
      iter = a_numIters-10;
    }
  }

  return m_mesh;
}