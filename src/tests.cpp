#include "tests.h"
#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <functional>

#include "LiteMath.h"
using namespace LiteMath;

#include <cassert>
#include <iomanip>

#include "dmesh.h"
#include "drender.h"
#include "scenes.h"
#include "optimizer.h"
#include "drender_mitsuba.h"

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

std::vector<CamInfo> create_cameras_around(int cam_num, int sensor_w, int sensor_h)
{
  std::vector<CamInfo> cameras;
  float4x4 mProj = LiteMath::perspectiveMatrix(45.0f, sensor_w/sensor_h, 0.1f, 100.0f);
  float rot_y = (2*M_PI)/cam_num;
  float rot_x = (M_PI_2)/cam_num;
  int st = cam_num/2;
  for(int i=0;i<cam_num;i++)
  {
    auto rot = LiteMath::rotate4x4Y(rot_y*(i-st))*LiteMath::rotate4x4Y(rot_x*(i-st));
    cameras.push_back(CamInfo(float3(-5*sin(rot_y*(i-st)),0,-5*cos(rot_y*(i-st))), float3(0,0,0), float3(0,1,0), sensor_w, sensor_h));
  }

  return cameras;
}

void optimization_test(const std::string &test_name,
                       std::function<void(TriangleMesh&, TriangleMesh&)> create_scene,
                       const DiffRenderSettings &diff_render_settings,
                       const OptimizerParameters &opt_parameters,
                       int opt_steps = 300,
                       bool test_by_steps = false,
                       int cameras_count = 3,
                       int image_w = 256,
                       int image_h = 256,
                       std::function<std::vector<CamInfo>(int, int, int)> create_cameras = create_cameras_around)
{
  auto cameras = create_cameras(cameras_count, image_w, image_h);

  Scene initialScene, targetScene;
  TriangleMesh initialMesh, targetMesh;
  create_scene(initialMesh, targetMesh);
  initialScene.add_mesh(initialMesh);
  targetScene.add_mesh(targetMesh);

  auto pDRender = MakeDifferentialRenderer(initialScene, diff_render_settings);

  Img img(image_w, image_h);
  Img targets[cameras_count];
  for (int i = 0; i < cameras_count; i++)
  {
    targets[i].resize(img.width(), img.height());
    targets[i].clear(float3{0, 0, 0});
  }

  pDRender->commit(targetScene);
  pDRender->render(targetScene, cameras.data(), targets, cameras_count);

  for(int i=0;i<cameras_count;i++) 
  {
    std::stringstream strOut;
    strOut  << "output/rendered_opt" << i << "/z_target.bmp";
    auto temp = strOut.str();
    LiteImage::SaveImage(temp.c_str(), targets[i]);
  }

  IOptimizer *pOpt = CreateSimpleOptimizer();
  float error = 1e9;
  if (!test_by_steps)
  {
    pOpt->Init(initialScene, pDRender, cameras.data(), targets, cameras_count, opt_parameters);
    Scene res_scene = pOpt->Run(opt_steps, error);
  }
  else
  {
    std::vector<Scene> iter_scenes;
    pOpt->Init(initialScene, pDRender, cameras.data(), targets, cameras_count, opt_parameters);
    Scene res_scene = pOpt->Run(opt_steps, error, &iter_scenes);
    for (auto &s : iter_scenes)
    {
      auto res = Tester::test_derivatives(s, targetScene, cameras[2], diff_render_settings, 100, 0);
      printf("derivatives error %f %f\n",res.pos_error, res.texture_error);      
    }
  }

  float psnr = -10*log10(max(1e-9f,error));
  bool pass = psnr > 40;
  printf("%s %s with PSNR %.3f\n", pass ? "    PASSED:" : "FAILED:    ", test_name.c_str(), psnr);
}

void Tester::test_2_1_triangle()
{
  optimization_test("TEST 2.1: ONE TRIANGLE SHAPE OPTIMIZATION",
                    scn03_Triangle3D_White,
                    {SHADING_MODEL::SILHOUETTE, 4},
                    {OptimizerParameters::GD_Adam, 0.2, 0.1},
                    300);
}

void Tester::test_2_2_pyramid()
{
  optimization_test("TEST 2.2: PYRAMID SHAPE OPTIMIZATION",
                    scn05_Pyramid3D,
                    {SHADING_MODEL::SILHOUETTE, 4},
                    {OptimizerParameters::GD_Adam, 0.05, 0.1},
                    300);
}

void Tester::test_2_3_sphere()
{
  optimization_test("TEST 2.3: SPHERE SHAPE OPTIMIZATION",
                    scn09_Sphere3D_Textured,
                    {SHADING_MODEL::SILHOUETTE, 4},
                    {OptimizerParameters::GD_Adam, 0.05, 0.1},
                    300);
}

void Tester::test_2_4_pyramid_vcol()
{
  optimization_test("TEST 2.4: PYRAMID VCOL+POS OPTIMIZATION",
                    scn05_Pyramid3D,
                    {SHADING_MODEL::VERTEX_COLOR, 4},
                    {OptimizerParameters::GD_Adam, 0.05, 0.0},
                    300);
}

void Tester::test_2_5_teapot_diffuse()
{
  optimization_test("TEST 2.5: TEAPOT DIFFUSE TEXTURE OPTIMIZATION",
                    scn10_Teapot3D_Textured,
                    {SHADING_MODEL::DIFFUSE, 16},
                    {OptimizerParameters::GD_Adam, 0.0, 0.05},
                    300);
}

void mitsuba_compare_test(const std::string &test_name,
                          std::function<void(TriangleMesh&, TriangleMesh&)> create_scene,
                          const DiffRenderSettings &diff_render_settings,
                          int cameras_count = 3,
                          int image_w = 256,
                          int image_h = 256,
                          std::function<std::vector<CamInfo>(int, int, int)> create_cameras = create_cameras_around)
{
  auto cameras = create_cameras(cameras_count, image_w, image_h);

  Scene initialScene, targetScene;
  TriangleMesh initialMesh, targetMesh;
  create_scene(initialMesh, targetMesh);
  initialScene.add_mesh(initialMesh);
  targetScene.add_mesh(targetMesh);
  initialScene.transform_meshes(true, true, true);
  targetScene.transform_meshes(true, true, true);

  auto pDRender = new DiffRenderMitsuba();
  pDRender->init(diff_render_settings);
  auto pDRenderOurs = MakeDifferentialRenderer(initialScene, diff_render_settings);

  Img img(image_w, image_h);
  Img targets[cameras_count];
  Img targetsOurs[cameras_count];
  for (int i = 0; i < cameras_count; i++)
  {
    targets[i].resize(img.width(), img.height());
    targets[i].clear(float3{0, 0, 0});
    targetsOurs[i].resize(img.width(), img.height());
    targetsOurs[i].clear(float3{0, 0, 0});
  }

  DTriangleMesh gradMeshMitsuba;
  {
    pDRender->commit(targetScene);
    pDRender->render(targetScene, cameras.data(), targets, cameras_count);
    gradMeshMitsuba.reset(initialScene.get_mesh(0), pDRender->mode);
    pDRender->d_render_and_compare(initialScene, cameras.data(), targets, cameras_count, img.width()*img.height(), gradMeshMitsuba);
  }

  DTriangleMesh gradMeshOurs;
  {
    pDRenderOurs->commit(targetScene);
    pDRenderOurs->render(targetScene, cameras.data(), targetsOurs, cameras_count);
    gradMeshOurs.reset(initialScene.get_mesh(0), pDRenderOurs->mode);
    pDRenderOurs->d_render_and_compare(initialScene, cameras.data(), targetsOurs, cameras_count, img.width()*img.height(), gradMeshOurs);
  }

  assert(gradMeshMitsuba.size() == gradMeshOurs.size());
  int sz = gradMeshOurs.size();
  float diff = 0;
  double acc1 = 1e-12;
  double acc2 = 1e-12;
  for (int i=0;i<sz;i++)
    acc1 += gradMeshMitsuba[i];
  for (int i=0;i<sz;i++)
    acc2 += gradMeshOurs[i];
  for (int i=0;i<sz;i++)
  {
    //logerr("[%d]%f %f",i, sz*gradMeshMitsuba[i]/acc1, sz*gradMeshOurs[i]/acc2);
    diff += abs(gradMeshMitsuba[i]/acc1 - gradMeshOurs[i]/acc2);
  }

  double image_diff = 0.0;
  for (int i=0;i<cameras_count;i++)
  {
    std::string path1 = "output/t"+std::to_string(i)+"_1.png";
    std::string path2 = "output/t"+std::to_string(i)+"_2.png";
    LiteImage::SaveImage(path1.c_str(), targets[i]);
    LiteImage::SaveImage(path2.c_str(), targetsOurs[i]);
    image_diff += LossAndDiffLoss(targets[i], targetsOurs[i], img);
  }
  image_diff /= cameras_count*img.width()*img.height();
  float psnr = -10*log10(max(1e-9,image_diff));
  diff /= sz;
  delete pDRender;
    
  bool pass = diff < 0.05;
  printf("%s %s with image PSNR %.5f\n", (psnr > 35) ? "    PASSED:" : "FAILED:    ", test_name.c_str(), psnr);
  printf("%s %s with derivatives difference %.5f\n", pass ? "    PASSED:" : "FAILED:    ", test_name.c_str(), diff);
}

void Tester::test_3_1_mitsuba_triangle()
{
  mitsuba_compare_test("TEST 3.1: TRIANGLE MITSUBA COMPARE",
                       scn03_Triangle3D_White,
                       {SHADING_MODEL::SILHOUETTE, 16},
                       1,
                       512,
                       512);
}

void Tester::test_3_2_mitsuba_sphere()
{
  mitsuba_compare_test("TEST 3.2: SPHERE MITSUBA COMPARE",
                       scn09_Sphere3D_Textured,
                       {SHADING_MODEL::SILHOUETTE, 16},
                       1,
                       512,
                       512);
}

void Tester::test_3_3_mitsuba_teapot()
{
  mitsuba_compare_test("TEST 3.3: TEAPOT MITSUBA COMPARE",
                       scn11_Teapot3D_Textured,
                       {SHADING_MODEL::SILHOUETTE, 16},
                       1,
                       512,
                       512);
}

void Tester::test_3_4_mitsuba_cube()
{
  mitsuba_compare_test("TEST 3.4: CUBE MITSUBA COMPARE",
                       scn06_Cube3D_VColor,
                       {SHADING_MODEL::SILHOUETTE, 16},
                       3,
                       512,
                       512);
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

    if (mesh.colors.size() == mesh.vertices.size() && a_pDRImpl->mode == SHADING_MODEL::VERTEX_COLOR)
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