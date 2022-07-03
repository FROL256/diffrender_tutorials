cmake_minimum_required (VERSION 3.8)
project (diffrender)

option(COMPLEX_OPT "Use Complex Optimizer" OFF)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

if(COMPLEX_OPT)
  add_definitions(-DCOMPLEX_OPT)
  include_directories("/usr/local/include/eigen3")
endif()

set (CMAKE_CXX_STANDARD 14)
include_directories(".", "include", "LiteMath")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

if(COMPLEX_OPT)
add_executable(drender 2d_triangles.cpp
                       fin_diff.cpp 
                       scenes.cpp
                       qmc.cpp
                       optimizer_simple.cpp
                       optimizer_complex.cpp
                       LiteMath/Image2d.cpp)


else()
add_executable(drender 2d_triangles.cpp
                       fin_diff.cpp
                       scenes.cpp
                       qmc.cpp 
                       optimizer_simple.cpp
                       LiteMath/Image2d.cpp)
endif()