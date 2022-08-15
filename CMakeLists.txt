cmake_minimum_required (VERSION 3.8)
project (diffrender)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 14)
include_directories(".", "include", "LiteMath")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

find_package(OpenMP REQUIRED)
#et(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

add_executable(drender main.cpp
                       drender.cpp
                       functions.cpp
                       fin_diff.cpp
                       scenes.cpp
                       qmc.cpp 
                       raytrace2d.cpp
                       raytrace3d.cpp
                       optimizer_simple.cpp
                       LiteMath/Image2d.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(drender PUBLIC OpenMP::OpenMP_CXX)
endif()