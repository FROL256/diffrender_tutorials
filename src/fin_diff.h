#pragma once
#include "utils.h"

struct DTriangleMesh;
struct TriangleMesh;
struct Scene;

void PrintAndCompareGradients(const DTriangleMesh& grad1, const DTriangleMesh& grad2);
void d_finDiff(const Scene &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);


void d_finDiff2(const Scene &mesh, const char* outFolder, const Img& origin, const Img& target, std::shared_ptr<IDiffRender> a_pDRImpl, const CamInfo& a_camData,
               DTriangleMesh &d_mesh, float dPos = 1.0f, float dCol = 0.01f);