#pragma once
#include "dmodels.h"

std::vector<float> sample_bilinear_clamp(float2 tc, const CPUTexture &tex);