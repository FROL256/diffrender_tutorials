#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <vector>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

#include "Bitmap.h"

float MSE_RGB_LDR(const std::vector<unsigned>& image1, const std::vector<unsigned>& image2)
{
  if(image1.size() != image2.size())
    return 10000000.0f;

  double accum = 0.0;

  for (size_t i = 0; i < image1.size(); ++i)
  {
    const unsigned pxData1 = image1[i];
    const unsigned pxData2 = image2[i];
    const unsigned r1      = (pxData1 & 0x00FF0000) >> 16;
    const unsigned g1      = (pxData1 & 0x0000FF00) >> 8;
    const unsigned b1      = (pxData1 & 0x000000FF);
     
    const unsigned r2      = (pxData2 & 0x00FF0000) >> 16;
    const unsigned g2      = (pxData2 & 0x0000FF00) >> 8;
    const unsigned b2      = (pxData2 & 0x000000FF);

    const float diffR = float(r1)-float(r2);
    const float diffG = float(b1)-float(b2);
    const float diffB = float(g1)-float(g2);

    accum += double(diffR * diffR + diffG * diffG + diffB * diffB);
  }

  return float(accum / double(image1.size()));
}


int main(int argc, const char** argv)
{
  for (const auto & entry : fs::directory_iterator("."))
  {
    std::string path = entry.path();
    if(path.substr(path.find_last_of(".") + 1) == "bmp")
    {  
      std::string path2 = "./reference/" + path.substr(2); 
      std::cout << path.c_str() << " -- " << path2.c_str();

      int w,h,w2,h2;
      auto image1 = LoadBMP(path.c_str(), &w, &h);
      auto image2 = LoadBMP(path2.c_str(), &w2, &h2);
      
      float mse = MSE_RGB_LDR(image1, image2);
      std::cout << " : " << mse << std::endl;
    }
  }
  return 0;    
}

