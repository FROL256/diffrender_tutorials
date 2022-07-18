#include "optimizer.h"

#include <random>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace LiteMath;

struct OptComplex : public IOptimizer
{
  OptComplex(){}

  void         Init(const TriangleMesh& a_mesh, const Img& a_image) override;
  TriangleMesh Run (size_t a_numIters = 100) override;

  TriangleMesh m_mesh; ///<! global mesh optimized mesh
  Img          m_targetImage;
  size_t       m_iter = 0;
};

IOptimizer* CreateComplexOptimizer() { return new OptComplex; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Eigen/Dense>              // optimization methods
#define OPTIM_ENABLE_EIGEN_WRAPPERS // optimization methods
#include "optim.hpp"                // optimization methods

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> EVector;

EVector VectorFromMesh(const TriangleMesh& a_mesh)
{
  EVector result( a_mesh.vertices.size()*3 + a_mesh.colors.size()*3 );
  size_t currPos = 0;
  for(size_t vertId=0; vertId< a_mesh.vertices.size(); vertId++, currPos+=3)
  {
    result[currPos+0] = a_mesh.vertices[vertId].x;
    result[currPos+1] = a_mesh.vertices[vertId].y;
    result[currPos+2] = a_mesh.vertices[vertId].z;
  }
  
  const size_t colorsNum = (a_mesh.m_meshType == TRIANGLE_FACE_COL) ? a_mesh.colors.size() : a_mesh.vertices.size();
  for(size_t colorId=0; colorId < colorsNum; colorId++, currPos+=3)
  {
    result[currPos+0] = a_mesh.colors[colorId].x;
    result[currPos+1] = a_mesh.colors[colorId].y;
    result[currPos+2] = a_mesh.colors[colorId].z;
  }

  return result;
}

float g_alphaPos   = 0.1f;
float g_alphaColor = 0.00001f;

EVector VectorFromDMesh(const DTriangleMesh& a_mesh)
{
  EVector result(a_mesh.totalParams());
  for(size_t i=0;i<result.size();i++)
    result[i] = a_mesh[i];
  return result;
}

TriangleMesh MeshFromVector(const EVector& a_vec, const TriangleMesh& a_mesh)
{
  TriangleMesh result = a_mesh;
  size_t currPos = 0;
  for(size_t vertId=0; vertId< result.vertices.size(); vertId++, currPos+=3)
  {
    result.vertices[vertId].x = a_vec[currPos+0];
    result.vertices[vertId].y = a_vec[currPos+1];
    result.vertices[vertId].z = a_vec[currPos+2];
  }
  const size_t colorsNum = (a_mesh.m_meshType == TRIANGLE_FACE_COL) ? a_mesh.colors.size() : a_mesh.vertices.size();
  for(size_t colorId=0; colorId < colorsNum; colorId++, currPos+=3)
  {
    result.colors[colorId].x = a_vec[currPos+0];
    result.colors[colorId].y = a_vec[currPos+1];
    result.colors[colorId].z = a_vec[currPos+2];
  }
  return result;
}

float EvalFunction(const EVector& vals_inp, EVector* grad_out, void* opt_data)
{
  OptComplex* pObj = (OptComplex*)opt_data;

  TriangleMesh mesh = MeshFromVector(vals_inp, pObj->m_mesh);
  
  constexpr int samples_per_pixel = 4;

  Img img(256, 256);
  render(mesh, samples_per_pixel, img);
  
  std::stringstream strOut;
  strOut  << "rendered_opt/render_" << std::setfill('0') << std::setw(4) << pObj->m_iter << ".bmp";
  auto tempStr = strOut.str();
  LiteImage::SaveImage(tempStr.c_str(), img);

  Img adjoint(img.width(), img.height(), float3{0, 0, 0});
  float mse = LossAndDiffLoss(img, pObj->m_targetImage, adjoint);
  
  DTriangleMesh d_mesh(mesh.vertices.size(), mesh.colors.size());
  d_mesh.clear();
  d_render(mesh, adjoint, samples_per_pixel, img.width() * img.height(), nullptr, nullptr, 
           d_mesh);

  
  std::cout << "iter " << pObj->m_iter << ", error = " << mse << std::endl;
  (*grad_out) = VectorFromDMesh(d_mesh); // apply 2.0f*summ(I[x,y] - I_target[x,y]) to get correct gradient for target image
  pObj->m_iter++;
  return mse;
}

void OptComplex::Init(const TriangleMesh& a_mesh, const Img& a_image) 
{ 
  m_mesh        = a_mesh; 
  m_targetImage = a_image; 
  m_iter        = 0; 
}

TriangleMesh OptComplex::Run(size_t a_numIters) 
{ 
  optim::algo_settings_t settings;
  settings.iter_max = a_numIters;
  settings.gd_settings.method             = 0; // 0 for simple gradient descend, 6 ADAM
  settings.gd_settings.par_step_size      = 1.0; // initialization for ADAM
  settings.gd_settings.step_decay         = true;
  settings.gd_settings.step_decay_periods = a_numIters/40;
  settings.gd_settings.step_decay_val     = 0.75f;
  settings.opt_error_value                = 20.0f;

  //g_alphaPos   = 0.3f; // for the first scene
  g_alphaPos   = 0.2f; // for the second scene
  g_alphaColor = 4.0f/float(m_targetImage.width()*m_targetImage.height()); 

  EVector x = VectorFromMesh(m_mesh);
  bool success = optim::gd(x, &EvalFunction, this, settings);
  std::cout << "OptComplex, optimization is FINISHED!" << std::endl;

  return MeshFromVector(x, m_mesh);
}