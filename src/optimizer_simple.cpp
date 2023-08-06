#include "optimizer.h"
#include "utils.h"
#include "scene.h"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace LiteMath;

void OptimizerParameters::set_default()
{
  if (alg == OPT_ALGORITHM::GD_Adam)
  {
    decayPeriod = 100;
    decay_mult = 0.75;
    base_lr = 0.1;
    position_lr = 0.1;
    textures_lr = 0.1;
  }
}

struct OptSimple : public IOptimizer
{
  OptSimple(){}

  void         Init(const Scene& a_scene, std::shared_ptr<IDiffRender> a_pDRImpl, 
                    const CamInfo* a_cams, const Img* a_images, int a_numViews, OptimizerParameters a_params) override;

  Scene Run (size_t a_numIters, float &final_error, std::vector<Scene> *iterations_dump = nullptr) override;

protected:
  
  typedef std::vector<std::pair<int, float>> IntervalLearningRate; //offset, learning rate

  float EvalFunction(const Scene& mesh, DTriangleMesh& gradMesh);
  IntervalLearningRate GetLR(DTriangleMesh& gradMesh);
  void  OptStep(DTriangleMesh &gradMesh, const IntervalLearningRate &lr);
  void  OptUpdateScene(const DTriangleMesh &gradMesh, Scene* mesh);
  void  StepDecay(int a_iterId, IntervalLearningRate &lr) const;

  Scene        m_scene; ///<! global mesh optimized mesh
  size_t       m_iter = 0;
  OptimizerParameters m_params;

  std::vector<GradReal> m_GSquare; ///<! m_GSquare is a vector of the sum of the squared gradients at or before iteration 'i'
  std::vector<GradReal> m_vec; 

  std::shared_ptr<IDiffRender> m_pDR = nullptr;
  const Img*     m_targets  = nullptr; 
  const CamInfo* m_cams     = nullptr; 
  int            m_numViews = 0;
};

IOptimizer* CreateSimpleOptimizer() { return new OptSimple; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OptSimple::IntervalLearningRate OptSimple::GetLR(DTriangleMesh& gradMesh)
{
  IntervalLearningRate lr;
  lr.push_back({0, m_params.position_lr});
  lr.push_back({gradMesh.color_offs(), m_params.base_lr});
  if (gradMesh.tex_count() > 0)
    lr.push_back({gradMesh.tex_offset(0), m_params.textures_lr});
  return lr;
}

void  OptSimple::StepDecay(int a_iterId, IntervalLearningRate &lr) const
{
  if(a_iterId > 0 && a_iterId % m_params.decayPeriod == 0)
  {
    for (auto &p : lr)
      p.second *= m_params.decay_mult;
  }
}

void OptSimple::OptUpdateScene(const DTriangleMesh &gradMesh, Scene* scene)
{
  auto &mesh = scene->meshes[0];//TODO: support multiple meshes
    for(int vertId=0; vertId< mesh.vertex_count(); vertId++)
    {
      mesh.vertices[vertId].x -= gradMesh.vertices_s()[3*vertId+0];
      mesh.vertices[vertId].y -= gradMesh.vertices_s()[3*vertId+1];
      mesh.vertices[vertId].z -= gradMesh.vertices_s()[3*vertId+2];
    }
    
    if (gradMesh.get_shading_model() == SHADING_MODEL::VERTEX_COLOR)
    {
      for(int faceId=0; faceId < mesh.colors.size(); faceId++)
      {
        mesh.colors[faceId].x -= gradMesh.colors_s()[3*faceId+0];
        mesh.colors[faceId].y -= gradMesh.colors_s()[3*faceId+1];
        mesh.colors[faceId].z -= gradMesh.colors_s()[3*faceId+2];
      }
    }
    
    if (gradMesh.get_shading_model() != SHADING_MODEL::SILHOUETTE &&
        gradMesh.get_shading_model() != SHADING_MODEL::VERTEX_COLOR)
    {
      for (int i=0;i<gradMesh.tex_count();i++)
      {
        int sz = mesh.textures[i].data.size();
        int off = gradMesh.tex_offset(i);
        for (int j=0;j<sz;j++)
          mesh.textures[i].data[j] -= gradMesh[off + j];
      }
    }
}

void OptSimple::OptStep(DTriangleMesh &gradMesh, const IntervalLearningRate &lr)
{
  if(m_params.alg >= OptimizerParameters::GD_AdaGrad)
  {
    if(m_params.alg == OptimizerParameters::GD_AdaGrad)  // ==> GSquare[i] = gradF[i]*gradF[i]
    {
      for(size_t i=0;i<gradMesh.size();i++)
        m_GSquare[i] += (gradMesh[i]*gradMesh[i]);
    }
    else if(m_params.alg == OptimizerParameters::GD_RMSProp) // ==> GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*gradF[i]*gradF[i]
    {
      const float alpha = 0.5f;
      for(size_t i=0;i<gradMesh.size();i++)
        m_GSquare[i] = 2.0f*(m_GSquare[i]*alpha + (gradMesh[i]*gradMesh[i])*(1.0f-alpha)); // does not works without 2.0f
    }
    else if(m_params.alg == OptimizerParameters::GD_Adam) // ==> Adam(m[i] = b*mPrev[i] + (1-b)*gradF[i], GSquare[i] = GSquarePrev[i]*a + (1.0f-a)*gradF[i]*gradF[i])
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
    for (int i=0;i<m_vec.size();i++)
      gradMesh[i] = gradMesh[i]/( std::sqrt(m_GSquare[i] + GradReal(1e-8f)));
  }
  
  for (int i=0;i<lr.size();i++)
  {
    int next_offset = (i == lr.size() - 1) ? m_vec.size() : lr[i+1].first;
    for (int j=lr[i].first; j<next_offset; j++)
      gradMesh[j] *= lr[i].second;
  }
}

float OptSimple::EvalFunction(const Scene& scene, DTriangleMesh& gradMesh)
{
  std::vector<Img> images(m_numViews);
  float mse = m_pDR->d_render_and_compare(scene, m_cams, m_targets, m_numViews, m_targets[0].width()*m_targets[0].height(), gradMesh, 
                                          m_params.verbose ? images.data() : nullptr);
  if (m_params.verbose)
  {
    for(int i=0;i<m_numViews;i++) 
    {
      std::stringstream strOut;
      strOut  << "rendered_opt" << i << "/render_" << std::setfill('0') << std::setw(4) << m_iter << ".bmp";
      auto temp = strOut.str();
      LiteImage::SaveImage(temp.c_str(), images[i]);
    }
  }
  m_iter++;
  return mse;
}

void OptSimple::Init(const Scene& a_scene, std::shared_ptr<IDiffRender> a_pDRImpl, 
                     const CamInfo* a_cams, const Img* a_images, int a_numViews, OptimizerParameters a_params) 
{ 
  assert(a_numViews > 0);

  m_scene       = a_scene; 
  m_iter        = 0; 
  m_pDR         = a_pDRImpl;

  m_targets     = a_images;
  m_cams        = a_cams;
  m_numViews    = a_numViews;
  m_params      = a_params;
}

Scene OptSimple::Run(size_t a_numIters, float &final_error, std::vector<Scene> *iterations_dump) 
{ 
  DTriangleMesh gradMesh;
  gradMesh.reset(m_scene.get_mesh(0), m_pDR->mode);//TODO: support multiple meshes

  if(m_params.alg >= OptimizerParameters::GD_AdaGrad) {
    m_GSquare.resize(gradMesh.size());
    memset(m_GSquare.data(), 0, sizeof(GradReal)*m_GSquare.size());
  }

  if(m_params.alg == OptimizerParameters::GD_Adam) {
    m_vec.resize(gradMesh.size());
    memset(m_vec.data(), 0, sizeof(GradReal)*m_vec.size());
  }

  auto lr = GetLR(gradMesh);
  
  m_iter = 0;
  float error = 0;
  float psnr = 0;

  for(size_t iter=0, trueIter = 0; iter < a_numIters; iter++, trueIter++)
  {
    error = EvalFunction(m_scene, gradMesh);
    psnr = -10*log10(max(1e-9f,error));
    if (m_params.verbose)
      std::cout << "iter " << trueIter << ", PSNR = " << psnr << std::endl;
    //PrintMesh(gradMesh);
    OptStep(gradMesh, lr);
    OptUpdateScene(gradMesh, &m_scene);
    StepDecay(iter, lr);

    if(error <= 1e-6 && (iter < a_numIters-10)) // perform last 10 iterations and stop
    {
      if (m_params.verbose)
        std::cout << "----------------------------> stop by error, perform last 10 iterations: " << std::endl;
      iter = a_numIters-10;
    }
    if (iterations_dump)
      iterations_dump->push_back(m_scene);
  }

  final_error = error;
  return m_scene;
}