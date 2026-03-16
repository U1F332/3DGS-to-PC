cmake_minimum_required(VERSION 3.8...3.31)

if(WIN32)
    set(_gs2pc_msvc_bin_root
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin")
    set(_gs2pc_msvc_x64_cl "${_gs2pc_msvc_bin_root}/Hostx64/x64/cl.exe")

    if(EXISTS "${_gs2pc_msvc_x64_cl}")
        if(NOT DEFINED CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER "${_gs2pc_msvc_x64_cl}" CACHE FILEPATH "Preferred MSVC x64 C compiler" FORCE)
        endif()
        if(NOT DEFINED CMAKE_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "${_gs2pc_msvc_x64_cl}" CACHE FILEPATH "Preferred MSVC x64 CXX compiler" FORCE)
        endif()
        if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
            set(CMAKE_CUDA_HOST_COMPILER "${_gs2pc_msvc_x64_cl}" CACHE FILEPATH "CUDA host compiler" FORCE)
        endif()
    endif()
endif()

project(gs2pc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GS2PC_ENABLE_CUDA_RASTER "Enable native CUDA raster wrappers" OFF)
set(GS2PC_CUDA_MIN_SM "75" CACHE STRING "Minimum CUDA SM architecture to support (e.g. 75, 86, 89)")
option(GS2PC_FORCE_MIN_CUDA_ARCH "Force CMAKE_CUDA_ARCHITECTURES from GS2PC_CUDA_MIN_SM" ON)

function(gs2pc_configure_cuda_architectures)
    if(NOT GS2PC_CUDA_MIN_SM MATCHES "^[0-9]+$")
        message(FATAL_ERROR "GS2PC_CUDA_MIN_SM must be an integer SM value, e.g. 75, 86, 89")
    endif()

    if(GS2PC_CUDA_MIN_SM LESS 50)
        message(FATAL_ERROR "GS2PC_CUDA_MIN_SM must be >= 50")
    endif()

    if(NOT GS2PC_FORCE_MIN_CUDA_ARCH AND DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        message(STATUS "GS2PC: keeping existing CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        return()
    endif()

    set(CMAKE_CUDA_ARCHITECTURES
        "${GS2PC_CUDA_MIN_SM}-real;${GS2PC_CUDA_MIN_SM}-virtual"
        CACHE STRING "CUDA architectures for gs2pc native" FORCE)

    message(STATUS "GS2PC: CMAKE_CUDA_ARCHITECTURES set to ${CMAKE_CUDA_ARCHITECTURES} (minimum SM ${GS2PC_CUDA_MIN_SM})")
endfunction()

set(GS2PC_CUDA_RASTER_AVAILABLE OFF)
if(GS2PC_ENABLE_CUDA_RASTER)
    if(WIN32 AND MSVC)
        if(DEFINED CMAKE_CXX_COMPILER)
            message(STATUS "GS2PC: using CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
        endif()
        if(DEFINED CMAKE_CUDA_HOST_COMPILER)
            message(STATUS "GS2PC: using CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
        endif()
    endif()

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        gs2pc_configure_cuda_architectures()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_subdirectory(
            ${CMAKE_CURRENT_SOURCE_DIR}/../gaussian-pointcloud-rasterization
            ${CMAKE_CURRENT_BINARY_DIR}/gaussian-pointcloud-rasterization
        )
        set(GS2PC_CUDA_RASTER_AVAILABLE ON)
    else()
        message(WARNING "CUDA compiler unavailable; building cpp_native without raster CUDA integration")
    endif()
endif()

add_library(gs2pc_core
    src/cli/cli.cpp
    src/io/io.cpp
    src/camera/camera_loader.cpp
    src/raster/raster.cpp
    src/converter/converter.cpp
    src/exporter/exporter.cpp
)

target_include_directories(gs2pc_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(gs2pc_core PRIVATE /permissive-)
else()
    target_compile_options(gs2pc_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GS2PC_CUDA_RASTER_AVAILABLE)
    target_link_libraries(gs2pc_core PRIVATE CudaRasterizer CUDA::cudart)
    target_compile_definitions(gs2pc_core PRIVATE GS2PC_HAS_CUDA_RASTER=1)
endif()

add_executable(gs2pc_cli
    src/cli/main.cpp
)

target_link_libraries(gs2pc_cli PRIVATE gs2pc_core)

set(GS2PC_PORTABLE_INSTALL_DIR "." CACHE STRING "Install destination for portable runtime files")

install(TARGETS gs2pc_cli
    RUNTIME DESTINATION "${GS2PC_PORTABLE_INSTALL_DIR}")

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21")
    install(FILES $<TARGET_RUNTIME_DLLS:gs2pc_cli>
        DESTINATION "${GS2PC_PORTABLE_INSTALL_DIR}"
        OPTIONAL)
endif()

if(WIN32 AND MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${GS2PC_PORTABLE_INSTALL_DIR}")
    include(InstallRequiredSystemLibraries)
endif()

if(GS2PC_CUDA_RASTER_AVAILABLE AND DEFINED CUDAToolkit_BIN_DIR)
    file(GLOB GS2PC_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64*.dll")
    if(GS2PC_CUDART_DLLS)
        install(FILES ${GS2PC_CUDART_DLLS}
            DESTINATION "${GS2PC_PORTABLE_INSTALL_DIR}")
    endif()
endif()

find_program(GS2PC_CLANG_FORMAT_BIN
    NAMES clang-format
    HINTS
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/bin")
