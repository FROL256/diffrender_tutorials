#pragma once
#include "utils.h"

struct DTriangleMesh;
struct TriangleMesh;
struct Scene;
struct IDiffRender;
struct CamInfo;
struct DiffRenderSettings;

class Tester
{
public:
  struct DerivativesTestResults
  {
    double pos_error = 0;
    double color_error = 0;
    double texture_error = 0;
    double average_error = 0;
  };
  void test_base_derivatives();
  void test_optimization();


  DerivativesTestResults test_derivatives(const Scene &initial_scene, const Scene &target_scene, const CamInfo& a_camData, const DiffRenderSettings &settings, 
                                          int max_test_vertices = 100, int max_test_texels = 100);
  void test_fin_diff(const Scene &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
                     DTriangleMesh &d_mesh, int debug_mesh_id, int max_test_vertices, int max_test_texels,
                     std::vector<bool> &tested_mask);
  DerivativesTestResults PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2, const std::vector<bool> &tested_mask);
};