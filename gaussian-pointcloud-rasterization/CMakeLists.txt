# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(_gs2pc_raster_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(_gs2pc_raster_cuda_architectures "70;75;86")
endif()

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "${_gs2pc_raster_cuda_architectures}")

target_compile_options(CudaRasterizer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-w>)
target_compile_options(CudaRasterizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--disable-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=221>
)

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
