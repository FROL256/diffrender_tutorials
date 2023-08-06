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
  static void test_base_derivatives();
  static void test_2_1_triangle();
  static void test_2_2_pyramid();
  static void test_2_3_sphere();
  static void test_2_4_pyramid_vcol();
  static void test_2_5_teapot_diffuse();
  static void test_2_6_path_tracing();

  static void test_3_1_mitsuba_triangle();
  static void test_3_2_mitsuba_sphere();
  static void test_3_3_mitsuba_teapot();

  static DerivativesTestResults test_derivatives(const Scene &initial_scene, const Scene &target_scene, const CamInfo& a_camData, const DiffRenderSettings &settings, 
                                                 int max_test_vertices = 100, int max_test_texels = 100);
  static void test_fin_diff(const Scene &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
                            DTriangleMesh &d_mesh, int debug_mesh_id, int max_test_vertices, int max_test_texels,
                            std::vector<bool> &tested_mask);
  static DerivativesTestResults PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2, const std::vector<bool> &tested_mask);
};