cmake_minimum_required (VERSION 3.8)
project (diffrender)


option(USE_EMBREE "Use Embree RT" OFF)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 14)
include_directories(".", "include", "LiteMath", "raytrace_src")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

if(USE_EMBREE)
  add_definitions(-DUSE_EMBREE)
  include_directories(${CMAKE_SOURCE_DIR}/external)
  include_directories(${CMAKE_SOURCE_DIR}/external/embree)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    link_directories(${CMAKE_SOURCE_DIR}/external/embree/lib_win64)
    set(RT_EMBREE_LIBS embree3)
  else()
    find_package(Threads REQUIRED)
    link_directories(${CMAKE_SOURCE_DIR}/external/embree/lib)
    set(RT_EMBREE_LIBS embree3 embree_sse42 embree_avx embree_avx2 lexers simd sys tasking)
  endif()

  set(SOURCE_RT
      raytrace_src/raytrace2d.cpp
      raytrace_src/raytrace3d_bf.cpp
      raytrace_src/raytrace3d_embree.cpp
      raytrace_src/EmbreeRT.cpp)
else()

  set(SOURCE_RT
      raytrace_src/raytrace2d.cpp
      raytrace_src/raytrace3d_bf.cpp)

  set(RT_EMBREE_LIBS "")

endif()

find_package(OpenMP REQUIRED)

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

add_executable(drender main.cpp
                       drender.cpp
                       functions.cpp
                       fin_diff.cpp
                       scenes.cpp
                       qmc.cpp 
                       ${SOURCE_RT}
                       optimizer_simple.cpp
                       LiteMath/Image2d.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(drender PUBLIC OpenMP::OpenMP_CXX ${RT_EMBREE_LIBS})
endif()