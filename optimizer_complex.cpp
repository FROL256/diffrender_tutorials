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

  TriangleMesh g_mesh; ///<! global mesh optimized mesh
  Img          g_targetImage;
  size_t       g_iter = 0;
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
  EVector result( a_mesh.vertices.size()*2 + a_mesh.colors.size()*3 );
  size_t currPos = 0;
  for(size_t vertId=0; vertId< a_mesh.vertices.size(); vertId++, currPos+=2)
  {
    result[currPos+0] = a_mesh.vertices[vertId].x;
    result[currPos+1] = a_mesh.vertices[vertId].y;
  }
  for(size_t faceId=0; faceId < a_mesh.colors.size(); faceId++, currPos+=3)
  {
   result[currPos+0] = a_mesh.colors[faceId].x;
   result[currPos+1] = a_mesh.colors[faceId].y;
   result[currPos+2] = a_mesh.colors[faceId].z;
  }
  return result;
}

constexpr float alphaPos   = 0.1f;
constexpr float alphaColor = 0.00001f;

EVector VectorFromDMesh(const DTriangleMesh& a_mesh)
{
  EVector result(a_mesh.totalParams());
  size_t currPos = 0;
  for(int vertId=0; vertId< a_mesh.numVertices(); vertId++, currPos+=2)
  {
    result[currPos+0] = a_mesh.vertices()[vertId].x*alphaPos;
    result[currPos+1] = a_mesh.vertices()[vertId].y*alphaPos;
  }
  for(int faceId=0; faceId < a_mesh.numFaces(); faceId++, currPos+=3)
  {
    result[currPos+0] = a_mesh.faceColors()[faceId].x*alphaColor;
    result[currPos+1] = a_mesh.faceColors()[faceId].y*alphaColor;
    result[currPos+2] = a_mesh.faceColors()[faceId].z*alphaColor;
  }
  return result;
}

TriangleMesh MeshFromVector(const EVector& a_vec, const TriangleMesh& a_mesh)
{
  TriangleMesh result = a_mesh;
  size_t currPos = 0;
  for(size_t vertId=0; vertId< result.vertices.size(); vertId++, currPos+=2)
  {
    result.vertices[vertId].x = a_vec[currPos+0];
    result.vertices[vertId].y = a_vec[currPos+1];
  }
  for(size_t faceId=0; faceId < result.colors.size(); faceId++, currPos+=3)
  {
    result.colors[faceId].x = a_vec[currPos+0];
    result.colors[faceId].y = a_vec[currPos+1];
    result.colors[faceId].z = a_vec[currPos+2];
  }
  return result;
}

float EvalFunction(const EVector& vals_inp, EVector* grad_out, void* opt_data)
{
  OptComplex* pObj = (OptComplex*)opt_data;

  TriangleMesh mesh = MeshFromVector(vals_inp, pObj->g_mesh);
  
  constexpr int samples_per_pixel = 4;

  Img img(256, 256);
  std::mt19937 rng(1234);
  render(mesh, samples_per_pixel, rng, img);
  
  std::stringstream strOut;
  strOut  << "rendered_opt/render_" << std::setfill('0') << std::setw(4) << pObj->g_iter << ".bmp";
  save_img(img, strOut.str());

  Img adjoint(img.width, img.height, float3{1, 1, 1});
  float mse = MSEAndDiff(img, pObj->g_targetImage, adjoint);
  Img dx(img.width, img.height), dy(img.width, img.height); // actually not needed here
  
  DTriangleMesh d_mesh(mesh.vertices.size(), mesh.colors.size());
  d_render(mesh, adjoint, samples_per_pixel, img.width * img.height , rng, dx, dy, d_mesh);
  
  std::cout << "iter " << pObj->g_iter << ", error = " << mse << std::endl;
  (*grad_out) = VectorFromDMesh(d_mesh); // apply 2.0f*summ(I[x,y] - I_target[x,y]) to get correct gradient for target image
  pObj->g_iter++;
  return mse;
}

void OptComplex::Init(const TriangleMesh& a_mesh, const Img& a_image) 
{ 
  g_mesh        = a_mesh; 
  g_targetImage = a_image; 
  g_iter        = 0; 
}

TriangleMesh OptComplex::Run(size_t a_numIters) 
{ 
  optim::algo_settings_t settings;
  settings.iter_max = a_numIters;
  settings.gd_settings.method = 0; // 0 for simple gradient descend, 6 ADAM
  settings.gd_settings.par_step_size      = 1.0; // initialization for ADAM
  settings.gd_settings.step_decay         = true;
  settings.gd_settings.step_decay_periods = a_numIters/10;
  settings.gd_settings.step_decay_val     = 0.75f;
  settings.opt_error_value                = 20.0f;

  EVector x = VectorFromMesh(g_mesh);
  bool success = optim::gd(x, &EvalFunction, this, settings);
  std::cout << "OptComplex, optimization is FINISHED!" << std::endl;

  return MeshFromVector(x, g_mesh);
}