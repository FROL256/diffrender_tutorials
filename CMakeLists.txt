cmake_minimum_required (VERSION 3.8)
project (diffrender)

#default build
#cmake CMakeLists.txt -DDEBUG=OFF -DUSE_EMBREE=ON -DUSE_MITSUBA=ON

option(USE_EMBREE "Use Embree RT" OFF)
option(DEBUG "Debug mode" OFF)
option(USE_MITSUBA "Use Mitsuba 3" OFF)

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

if(USE_MITSUBA)
  find_package(PythonLibs REQUIRED)
  include_directories( ${PYTHON_INCLUDE_DIR} "external/python", "mitsuba_src")
  set(MITSUBA_LIBS ${PYTHON_LIBRARIES})
  set(SOURCE_MITSUBA
      mitsuba_src/drender_mitsuba.cpp
      mitsuba_src/mitsuba_python_interaction.cpp)
else()
  include_directories("mitsuba_src/stub")
  set(MITSUBA_LIBS "")
  set(SOURCE_MITSUBA )
endif()

find_package(OpenMP REQUIRED)

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp" "common_src/*.cpp" "external/python")

if (DEBUG)
set(CMAKE_CXX_FLAGS "-g -fsanitize=address -O1 -fno-omit-frame-pointer -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type")
else()
set(CMAKE_CXX_FLAGS "-O3 -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type")
endif()

add_executable(drender ${SOURCES}
                       ${SOURCE_RT}
                       ${SOURCE_MITSUBA})

if(OpenMP_CXX_FOUND)
  target_link_libraries(drender PUBLIC OpenMP::OpenMP_CXX ${RT_EMBREE_LIBS} ${MITSUBA_LIBS})
endif()