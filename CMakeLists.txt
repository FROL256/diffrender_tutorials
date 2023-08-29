cmake_minimum_required (VERSION 3.8)
project (diffrender)

#default build
#cmake CMakeLists.txt -DCMAKE_BUILD_TYPE=Release -DDEBUG=OFF -DUSE_EMBREE=ON -DUSE_OPENMP=ON -DUSE_MITSUBA=ON -DUSE_ENZYME=ON -DCLANG_VERSION=9
#cmake .. -DCMAKE_BUILD_TYPE=Release -DDEBUG=OFF -DUSE_EMBREE=ON -DUSE_OPENMP=ON -DUSE_MITSUBA=OFF -DUSE_ENZYME=ON -DCLANG_VERSION=15 
#
set(CLANG_VERSION "9" CACHE STRING "clang and LLVM version to use")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-${CLANG_VERSION}")
option(USE_EMBREE "Use Embree RT" OFF)
option(DEBUG "Debug mode" OFF)
option(USE_MITSUBA "Use Mitsuba 3" OFF)
option(USE_OPENMP "Use Open MP" ON)
option(USE_ENZYME "Use Enzyme" OFF)

if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(AUX_OMP_INC "/usr/lib/llvm-15/lib/clang/15.0.7/include/") # use 'locate omp.h'     on you system to find apropriate dir for clang
  set(AUX_OMP_LIB "/usr/lib/llvm-15/lib/")                      # use 'locate libomp.so' on you system to find apropriate dir for clang
endif()

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_CXX_STANDARD 14)
include_directories("src", "LiteMath", "raytrace_src", "common_src", "external/enzyme")

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

set(OMP_LIBS "")
if(OpenMP_CXX_FOUND)
  add_definitions(-DUSE_OMP)
  include_directories(${AUX_OMP_INC})
  link_directories(${AUX_OMP_LIB})
  set(OMP_LIBS OpenMP::OpenMP_CXX)
endif()

if (USE_ENZYME)
set(SOURCE_ENZYME "")
else()
set(SOURCE_ENZYME external/enzyme/enzyme.cpp)
endif()
#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp" "common_src/*.cpp" "external/python")

if (DEBUG)
set(CMAKE_CXX_FLAGS "-g -fsanitize=address -O1 -fno-omit-frame-pointer -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type -Wno-unused-parameter -Wshadow")
else()
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} -O3 -W -Wno-sign-compare -Wno-int-to-pointer-cast -Wno-register -Wreturn-type -Wno-unused-parameter -Wshadow")
endif()

add_executable(drender ${SOURCES}
                       ${SOURCE_RT}
                       ${SOURCE_MITSUBA}
                       ${SOURCE_ENZYME})

target_link_libraries(drender PUBLIC ${OMP_LIBS} ${RT_EMBREE_LIBS} ${MITSUBA_LIBS})

if (USE_ENZYME)
  set(ENZYME_FLAGS "-fplugin=${CMAKE_SOURCE_DIR}/external/enzyme/enzyme/build/Enzyme/ClangEnzyme-${CLANG_VERSION}.so")
  message(${ENZYME_FLAGS})
  target_compile_options(drender PUBLIC ${ENZYME_FLAGS})
else()
endif()