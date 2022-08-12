#pragma once
#include "raytrace.h"

static inline void VertexShader(const CamInfo& u, float vx, float vy, float vz, 
                                float output[2])
{
  const float W    =   vx * u.projM[3] + vy * u.projM[7] + vz * u.projM[11] + u.projM[15]; 
  const float xNDC =  (vx * u.projM[0] + vy * u.projM[4] + vz * u.projM[ 8] + u.projM[12])/W;
  const float yNDC = -(vx * u.projM[1] + vy * u.projM[5] + vz * u.projM[ 9] + u.projM[13])/W;
  output[0] = (xNDC*0.5f + 0.5f)*u.width;
  output[1] = (yNDC*0.5f + 0.5f)*u.height; 
}

static inline float VS_X(float V[3], const CamInfo& data) // same as VertexShader().x
{
  const float W    =  V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = (V[0] * data.projM[0] + V[1] * data.projM[4] + V[2] * data.projM[ 8] + data.projM[12])/W;
  return (xNDC*0.5f + 0.5f)*data.width;
}

static inline float VS_Y(float V[3], const CamInfo& data) // // same as VertexShader().y
{
  const float W    =   V[0] * data.projM[3] + V[1] * data.projM[7] + V[2] * data.projM[11] + data.projM[15]; 
  const float xNDC = -(V[0] * data.projM[1] + V[1] * data.projM[5] + V[2] * data.projM[ 9] + data.projM[13])/W;
  return (xNDC*0.5f + 0.5f)*data.height;
}

void VS_X_grad(float V[3], const CamInfo &data, float _d_V[3]);
void VS_Y_grad(float V[3], const CamInfo &data, float _d_V[3]);

void BarU_grad(const float ray_pos[3], const float ray_dir[3], const float A[3], const float B[3], const float C[3], 
               float* _d_A, float* _d_B, float* _d_C);

void BarV_grad(const float ray_pos[3], const float ray_dir[3], const float A[3], const float B[3], const float C[3], 
               float* _d_A, float* _d_B, float* _d_C);

void BarW_grad(const float ray_pos[3], const float ray_dir[3], const float A[3], const float B[3], const float C[3], 
               float* _d_A, float* _d_B, float* _d_C);
