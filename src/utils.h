#pragma once
#include <vector>
#include <memory>
#include "LiteMath.h"
#include "Image2d.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::int3;
using LiteMath::int4;

using LiteMath::clamp;
using LiteMath::normalize;

using Img = LiteImage::Image2D<float3>;

struct DTriangleMesh;

float LossAndDiffLoss(const Img& b, const Img& a, Img& a_outDiff);
void PrintMesh(const DTriangleMesh& a_mesh);

void CHECK_NaN(float f);
void CHECK_NaN(float3 f);
static inline float MSE(const Img& b, const Img& a) { return LiteImage::MSE(b,a); }
static inline float2 normal2D(const float2 &v) {return float2{-v.y, v.x};} 
static inline float  edgeFunction(float2 a, float2 b, float2 c) { return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x); }  // actuattly just a mixed product ... :)
