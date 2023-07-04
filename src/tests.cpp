#include "tests.h"
#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>

#include "LiteMath.h"
using namespace LiteMath;

#ifdef WIN32
  #include <direct.h>     // for windows mkdir
#else
  #include <sys/stat.h>   // for linux mkdir
  #include <sys/types.h>
#endif

#include <cassert>
#include <iomanip>

#include "dmesh.h"
#include "drender.h"
#include "scenes.h"
#include "optimizer.h"

void Tester::test_base_derivatives()
{

  constexpr int IMAGE_W = 256;
  constexpr int IMAGE_H = 256;
  constexpr int SILHOUETTE_SPP = 4;
  constexpr int BASE_SPP = 16;
  
  CamInfo camera;
  camera.width  = float(IMAGE_W);
  camera.height = float(IMAGE_H);
  camera.mWorldView = LiteMath::translate4x4(float3(0,0,-3));
  camera.mProj = LiteMath::perspectiveMatrix(45.0f, camera.width / camera.height, 0.1f, 100.0f);
  camera.commit();

  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn03_Triangle3D_White(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);
    auto res = test_derivatives(initialScene, targetScene, camera, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP}, 100, 0);

    bool pass = res.pos_error < 0.05;
    printf("%s TEST 1: EDGE SAMPLING TRIANGLE with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", res.pos_error);
  }

  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn05_Pyramid3D(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);
    auto res = test_derivatives(initialScene, targetScene, camera, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP}, 100, 0);

    bool pass = res.pos_error < 0.05;
    printf("%s TEST 2: EDGE SAMPLING PYRAMID with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", res.pos_error);
  }

  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn09_Sphere3D_Textured(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);
    auto res = test_derivatives(initialScene, targetScene, camera, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP}, 100, 0);

    bool pass = res.pos_error < 0.05;
    printf("%s TEST 3: EDGE SAMPLING SPHERE with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", res.pos_error);
  }

  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn05_Pyramid3D(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);
    auto res = test_derivatives(initialScene, targetScene, camera, {SHADING_MODEL::VERTEX_COLOR, SILHOUETTE_SPP}, 100, 0);

    bool pass = res.color_error < 0.05;
    printf("%s TEST 4: VCOLOR DERIVATIVES with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", res.color_error);
  }

  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn09_Sphere3D_Textured(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);
    auto res = test_derivatives(initialScene, targetScene, camera, {SHADING_MODEL::DIFFUSE, SILHOUETTE_SPP}, 0, 250);

    bool pass = res.texture_error < 0.05;
    printf("%s TEST 5: TEXTURE DERIVATIVES with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", res.texture_error);
  }
}

void Tester::test_optimization()
{
  constexpr int IMAGE_W = 256;
  constexpr int IMAGE_H = 256;
  constexpr int SILHOUETTE_SPP = 16;
  constexpr int BASE_SPP = 16;

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
  cameras[0].mWorldView = LiteMath::translate4x4(float3(0,0,-3));

  cameras[1].mProj      = mProj;
  cameras[1].mWorldView = LiteMath::translate4x4(float3(0,0,-3))*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*120.0f)*LiteMath::rotate4x4X(LiteMath::DEG_TO_RAD*45.0f);

  cameras[2].mProj      = mProj;
  cameras[2].mWorldView = LiteMath::translate4x4(float3(0,0,-3))*LiteMath::rotate4x4Y(LiteMath::DEG_TO_RAD*(-120.0f))*LiteMath::rotate4x4X(LiteMath::DEG_TO_RAD*(-45.0f));

  for(int i=0;i<camsNum;i++)
    cameras[i].commit();

  auto g_uniforms = cameras[0];

  if (true)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn03_Triangle3D_White(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.2;
    op.textures_lr = 0.0;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.1: ONE TRIANGLE SHAPE OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }

  if (true)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn05_Pyramid3D(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.2;
    op.textures_lr = 0.0;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.2: PYRAMID SHAPE OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }

  if (true)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn09_Sphere3D_Textured(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::SILHOUETTE, SILHOUETTE_SPP});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.2;
    op.textures_lr = 0.0;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.3: SPHERE SHAPE OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }

  if (false)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn05_Pyramid3D(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::VERTEX_COLOR, BASE_SPP});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.1;
    op.textures_lr = 0.0;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.4: VCOL+POS OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }

  if (true)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn10_Teapot3D_Textured(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::DIFFUSE, BASE_SPP});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.0;
    op.textures_lr = 0.2;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.5: DIFFUSE TEXTURE OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }

  if (true)
  {
    Scene initialScene, targetScene;
    TriangleMesh initialMesh, targetMesh;
    scn11_Teapot3D_Textured(initialMesh, targetMesh);
    initialScene.add_mesh(initialMesh);
    targetScene.add_mesh(targetMesh);

    auto pDRender = MakeDifferentialRenderer(initialScene, {SHADING_MODEL::PATH_TEST, 64});
    
    Img targets[camsNum];
    for(int i=0;i<camsNum;i++) 
    {
      targets[i].resize(img.width(), img.height());
      targets[i].clear(float3{0,0,0});
    }

    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras, targets, camsNum);

    IOptimizer* pOpt = CreateSimpleOptimizer();
    OptimizerParameters op = OptimizerParameters(OptimizerParameters::GD_Adam);
    op.position_lr = 0.0;
    op.textures_lr = 0.2;
    op.verbose = false;
    pOpt->Init(initialScene, pDRender, cameras, targets, 3, op);

    float error = 1e9;
    Scene res_scene = pOpt->Run(300, error);

    bool pass = error < 1;
    printf("%s TEST 2.6: PATH TRACING OPTIMIZATION with error %.3f\n", pass ? "    PASSED:" : "FAILED:    ", error);
  }
}

void finDiff_param(float *param, GradReal *param_diff, Img &out_diffImage, float delta,
                   Img &img, const Img &target, const Img &MSEOrigin, const CamInfo& a_camData, const Scene &copy_scene,
                   std::shared_ptr<IDiffRender> a_pDRImpl, bool geom_changed)
{
  *param += delta;
  
  if (geom_changed)
    a_pDRImpl->commit(copy_scene);
  a_pDRImpl->render(copy_scene, &a_camData, &img, 1);
    
  out_diffImage = (LiteImage::MSEImage(img,target) - MSEOrigin)/delta;   
  float3 summColor = SummOfPixels(out_diffImage); 
  *param_diff += GradReal(summColor.x + summColor.y + summColor.z);

  *param -= delta;
}

//cnt unique integers in range [from, to)
std::vector<int> random_unique_indices(int from, int to, int cnt)
{
  if (cnt <= 0)
    return {};
  std::vector<int> res;
  if (to - from <= cnt)
  {
    res.resize(to-from, 0);
    for (int i=from;i<to;i++)
      res[i-from] = i;
  }
  else
  {
    int sh_cnt = min(4*cnt, to - from);
    int step = (to - from)/sh_cnt;
    std::vector<int> sh;
    sh.reserve((to - from)/step);
    if (step == 1)
    {
      for (int i=from;i<to;i+=step)
        sh.push_back(i);
    }
    else
    {
      for (int i=from;i<to;i+=step)
        sh.push_back(i + rand()%step);
    }

    std::random_shuffle(sh.begin(), sh.end());
    res = std::vector<int>(sh.begin(), sh.begin() + cnt);
  }
  return res;
}

void Tester::test_fin_diff(const Scene &scene, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl,
                           const CamInfo& a_camData, DTriangleMesh &d_mesh, int debug_mesh_id, int max_test_vertices, int max_test_texels,
                           std::vector<bool> &tested_mask)
{
  float dPos = 2.0f/float(origin.width());
  float dCol = 0.01f;
  bool save_images = true;

  Scene copy_scene = scene;
  a_pDRImpl->commit(copy_scene);
  TriangleMesh &mesh = copy_scene.meshes[debug_mesh_id];
  d_mesh.reset(mesh, a_pDRImpl->mode);
  Img MSEOrigin = LiteImage::MSEImage(origin, target);
  Img img(origin.width(), origin.height());

  #define pFinDiff(param, dmesh_offset, out_image, delta, geom_changed) \
          tested_mask[(int)(dmesh_offset)] = true; finDiff_param(param, d_mesh.data() + (int)(dmesh_offset), out_image, delta, img, target, MSEOrigin, a_camData, copy_scene, a_pDRImpl, geom_changed);
  
  std::vector<int> debug_vertex_ids = random_unique_indices(0, mesh.vertex_count(), max_test_vertices);
  Img pos_x, pos_y, pos_z;
  for(auto &i : debug_vertex_ids)
  {
    pFinDiff(mesh.vertices[i].M + 0, d_mesh.vert_offs() + i*3+0, pos_x, dPos, true);
    pFinDiff(mesh.vertices[i].M + 1, d_mesh.vert_offs() + i*3+1, pos_y, dPos, true);
    pFinDiff(mesh.vertices[i].M + 2, d_mesh.vert_offs() + i*3+2, pos_z, dPos, true);

    if (save_images)
    {
      Img diffImage(pos_x.width(), pos_x.height()); 
      for(int y=0;y<pos_x.height();y++)
        for(int x=0;x<pos_x.width();x++)
          diffImage[int2(x,y)] = float3(pos_x[int2(x,y)].x, pos_y[int2(x,y)].x, pos_z[int2(x,y)].x);

      if(outFolder != nullptr)
      {
        std::stringstream strOut;
        strOut << outFolder << "/" << "pos_xyz_" << i << ".bmp";
        auto path = strOut.str();
        LiteImage::SaveImage(path.c_str(), diffImage);
      }
    }

    if (mesh.colors.size() == mesh.vertices.size())
    {
      pFinDiff(mesh.colors[i].M + 0, d_mesh.color_offs() + i*3+0, pos_x, dCol, true);
      pFinDiff(mesh.colors[i].M + 1, d_mesh.color_offs() + i*3+1, pos_y, dCol, true);
      pFinDiff(mesh.colors[i].M + 2, d_mesh.color_offs() + i*3+2, pos_z, dCol, true);

      if (save_images)
      {
        Img diffImage(pos_x.width(), pos_x.height()); 
        for(int y=0;y<pos_x.height();y++)
          for(int x=0;x<pos_x.width();x++)
            diffImage[int2(x,y)] = float3(pos_x[int2(x,y)].x, pos_y[int2(x,y)].x, pos_z[int2(x,y)].x);

        if(outFolder != nullptr)
        {
          std::stringstream strOut;
          strOut << outFolder << "/" << "color_xyz_" << i << ".bmp";
          auto path = strOut.str();
          LiteImage::SaveImage(path.c_str(), diffImage);
        }
      }
    }
  }

  for (int tex_n = 0; tex_n < mesh.textures.size(); tex_n++)
  {
    CPUTexture &tex = mesh.textures[tex_n];
    std::vector<int2> debug_texel_ids;
    std::vector<int> dtids = random_unique_indices(0, tex.w*tex.h, max_test_texels);
    for(auto ind : dtids)
    {
      debug_texel_ids.push_back(int2(ind % tex.w, ind / tex.w));
    }
    
    for(auto &i : debug_texel_ids)
    {
      for (int ch =0; ch < tex.channels; ch++)
      {
        int off = d_mesh.tex_offset(tex_n) + tex.pixel_to_offset(i.x, i.y) + ch;
        pFinDiff(tex.data.data() + tex.pixel_to_offset(i.x, i.y) + ch, off, pos_x, 0.1, false);
      }
    }
  }
}

Tester::DerivativesTestResults Tester::test_derivatives(const Scene &initial_scene, const Scene &target_scene, const CamInfo& a_camData, 
                                                        const DiffRenderSettings &settings, int max_test_vertices, int max_test_texels)
{
  Img original = Img(a_camData.width, a_camData.height);
  Img target = Img(a_camData.width, a_camData.height);
  Img tmp = Img(a_camData.width, a_camData.height);

  auto Render = MakeDifferentialRenderer(target_scene, settings);

  Render->commit(target_scene);
  Render->render(target_scene, &a_camData, &target, 1);

  Render->commit(initial_scene);
  Render->render(initial_scene, &a_camData, &original, 1);

  LossAndDiffLoss(original, target, tmp); 
  DerivativesTestResults r;
  for (int i=0;i<initial_scene.meshes.size();i++)
  {
    DTriangleMesh dMesh1 = DTriangleMesh(initial_scene.meshes[i], settings.mode);
    DTriangleMesh dMesh2 = DTriangleMesh(initial_scene.meshes[i], settings.mode);
    std::vector<bool> mask(dMesh1.size(), false);
    
    Render->commit(initial_scene);
    Render->d_render(initial_scene, &a_camData, &tmp, 1, target.width()*target.height(), dMesh1);
    test_fin_diff(initial_scene, nullptr, original, target, Render, a_camData, dMesh2, i, max_test_vertices, max_test_texels, mask);

    auto rm = PrintAndCompareGradients(dMesh1, dMesh2, mask);

    r.pos_error += rm.pos_error/initial_scene.meshes.size();
    r.color_error += rm.color_error/initial_scene.meshes.size();
    r.texture_error += rm.texture_error/initial_scene.meshes.size();
    r.average_error += rm.average_error/initial_scene.meshes.size();
  }

  return r;
}

Tester::DerivativesTestResults Tester::PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2, const std::vector<bool> &tested_mask)
{
  double posError = 0.0;
  double colError = 0.0;
  double texError = 0.0;
  double posLengthL1 = 1e-12;
  double colLengthL1 = 1e-12;
  double texLengthL1 = 1e-12;

  for(size_t i=0; i<grad1.color_offs(); i++) 
  {
    if (!tested_mask[i])
      continue;
    double diff = std::abs(double(grad1[i] - grad2[i]));
    posError    += diff;
    posLengthL1 += std::abs(grad1[i]) + std::abs(grad2[i]);
    //std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[i] << "\t";  
    //std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[i] << std::endl;
  }

  //std::cout << "--------------------------" << std::endl;
  for(size_t i=grad1.color_offs();i<grad1.textures_offset();i++) 
  {
    if (!tested_mask[i])
      continue;
    double diff = std::abs(double(grad1[i] - grad2[i]));
    colError   += diff;
    colLengthL1 += std::abs(grad1[i]) + std::abs(grad2[i]);
    //std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad1[i] << "\t";  
    //std::cout << std::fixed << std::setw(8) << std::setprecision(4) << grad2[i] << std::endl;
  }
  
  assert(grad1.size() == grad2.size());

  if (grad1.tex_count() > 0)
  {
    for (int i=grad1.textures_offset();i<grad1.size();i++)
    {
      if (!tested_mask[i])
        continue;
      double diff = std::abs(double(grad1[i] - grad2[i]));
      texError += diff;
      texLengthL1 += std::abs(grad1[i]) + std::abs(grad2[i]);
    }
  }

  double totalError = posError + colError + texError;
  double totalLengthL1 = texLengthL1 + posLengthL1 + colLengthL1;

  //std::cout << "==========================" << std::endl;
  //std::cout << "GradErr[L1](vpos   ) = " << std::setw(10) << std::setprecision(4) << posError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(posError/posLengthL1) << "%" << std::endl;
  //std::cout << "GradErr[L1](color  ) = " << std::setw(10) << std::setprecision(4) << colError/double(grad1.numVerts()*3)    << "\t which is \t" << 100.0*(colError/colLengthL1) << "%" << std::endl;
  //std::cout << "GradErr[L1](texture) = " << std::setw(10) << std::setprecision(4) << texError                               << "\t which is \t" << 100.0*(texError/texLengthL1) << "%" << std::endl;
  //std::cout << "GradErr[L1](average) = " << std::setw(10) << std::setprecision(4) << totalError/double(grad1.size())        << "\t which is \t" << 100.0*(totalError/totalLengthL1) << "%" << std::endl;

  return DerivativesTestResults{posError/posLengthL1, colError/colLengthL1, texError/texLengthL1, totalError/totalLengthL1};
}