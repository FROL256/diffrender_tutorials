cmake_minimum_required (VERSION 3.8)
project (diffrender)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 14)
include_directories(".")
include_directories("include")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

#find_package(OpenMP)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

add_executable(drender 2d_triangles.cpp 
                       optimizer_simple.cpp
                       Bitmap.cpp)