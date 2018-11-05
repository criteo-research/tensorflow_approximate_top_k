cmake_minimum_required(VERSION 3.12)
project(approximate_top_k)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
find_package(TensorFlow REQUIRED)

include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})
include_directories(annoy/src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -O3 -ffast-math -fno-associative-math -march=native -undefined dynamic_lookup -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=0" )

add_executable(approximate_top_k
        src/approximate_top_k.cc
        src/approximate_top_k_op.cc)
