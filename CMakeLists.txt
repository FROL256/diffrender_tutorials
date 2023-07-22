cmake_minimum_required (VERSION 3.8)
project (diffrender)


option(USE_EMBREE "Use Embree RT" OFF)
option(DEBUG "Debug mode" OFF)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_CXX_STANDARD 14)
include_directories("src", "LiteMath", "raytrace_src", "common_src")

if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

if(USE_EMBREE)
  add_definitions(-DUSE_EMBREE)
  include_directories(${CMAKE_SOURCE_DIR}/external)
  include_directories(${CMAKE_SOURCE_DIR}/external/embree/include)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    link_directories(${CMAKE_SOURCE_DIR}/external/embree/lib_win64)
    set(RT_EMBREE_LIBS embree3)
  else()
    find_package(Threads REQUIRED)
    link_directories(${CMAKE_SOURCE_DIR}/external/embree/lib)
    set(RT_EMBREE_LIBS embree3 tbb)
  endif()

  set(SOURCE_RT
      raytrace_src/raytrace3d_bf.cpp
      raytrace_src/raytrace3d_embree.cpp
      raytrace_src/EmbreeRT.cpp)
else()

  set(SOURCE_RT
      raytrace_src/raytrace3d_bf.cpp)

  set(RT_EMBREE_LIBS "")

endif()

find_package(OpenMP REQUIRED)

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp" "common_src/*.cpp")

if (DEBUG)
set(CMAKE_CXX_FLAGS "-g -O0 -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type")
else()
set(CMAKE_CXX_FLAGS "-O3 -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type")
endif()

add_executable(drender ${SOURCES}
                       ${SOURCE_RT})

if(OpenMP_CXX_FOUND)
  target_link_libraries(drender PUBLIC OpenMP::OpenMP_CXX ${RT_EMBREE_LIBS})
endif()