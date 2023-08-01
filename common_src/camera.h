#pragma once
#include "LiteMath.h"

struct CamInfo
{
  LiteMath::float4x4 mWorldView;
  LiteMath::float4x4 mProj;

  float mWVP[16]; // WorlViewProject := (mProj*(mView*mWorld))
  float width;
  float height;
  
  /**
  \brief make all needed internal computations, prepare for rendering
  */
  void commit()
  {
    LiteMath::float4x4 mTransform = mProj*mWorldView;
    memcpy(mWVP, (float*)&mTransform, 16*sizeof(float));
  }
};