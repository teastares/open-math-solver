cmake_minimum_required(VERSION 3.26)
project(solver_openblas_dependency LANGUAGES C)

get_filename_component(solver_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
file(READ "${solver_workspace_root}/dependencies.lock.json" dependency_lock)
string(JSON openblas_url GET "${dependency_lock}" openblas url)
string(JSON openblas_hash GET "${dependency_lock}" openblas sha256)
string(JSON openblas_version GET "${dependency_lock}" openblas version)
string(JSON openblas_patch_path GET "${dependency_lock}" openblas local_patches 0 path)
string(JSON openblas_patch_hash GET "${dependency_lock}" openblas local_patches 0 sha256)
set(openblas_patch_file "${solver_workspace_root}/${openblas_patch_path}")
file(SHA256 "${openblas_patch_file}" actual_openblas_patch_hash)
if(NOT actual_openblas_patch_hash STREQUAL openblas_patch_hash)
  message(FATAL_ERROR "OpenBLAS 本地补丁与依赖锁不一致")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${solver_workspace_root}/dependencies.lock.json" "${openblas_patch_file}")
set(SOLVER_DEPENDENCY_JOBS 4 CACHE STRING "依赖构建并发数")
set(solver_blas_install "${CMAKE_CURRENT_BINARY_DIR}/install")
file(TO_CMAKE_PATH "${CMAKE_MAKE_PROGRAM}" solver_blas_make_program)
set(solver_blas_arch_args -DDYNAMIC_ARCH=ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
  list(APPEND solver_blas_arch_args -DTARGET=CORE2)
endif()

# CHOLMOD 使用 Fortran ABI BLAS 和四精度 POTRF；C 翻译的 LAPACK 无需 Fortran 编译器。
# 因为 SuiteSparse 配置时需要探测符号，本依赖先于主项目配置完成安装。
include(ExternalProject)
ExternalProject_Add(solver_openblas_build
  URL "${openblas_url}" URL_HASH "SHA256=${openblas_hash}"
  DOWNLOAD_DIR "${solver_workspace_root}/.tools/downloads"
  DOWNLOAD_NAME "openblas-${openblas_version}.tar.gz" DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/openblas-${openblas_version}"
  # 校验值进入补丁命令，使已有 ExternalProject 的补丁戳随内容变化而失效。
  PATCH_COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=<SOURCE_DIR>"
    "-DEXPECTED_PATCH_SHA256=${openblas_patch_hash}" -P "${openblas_patch_file}"
  CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}" "-DCMAKE_ASM_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_MAKE_PROGRAM=${solver_blas_make_program}" -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${solver_blas_install}" -DCMAKE_INSTALL_LIBDIR=lib
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DBUILD_TESTING=OFF
    -DBUILD_WITHOUT_LAPACK=OFF -DBUILD_WITHOUT_LAPACKE=ON -DBUILD_WITHOUT_CBLAS=OFF
    -DONLY_CBLAS=OFF -DC_LAPACK=ON -DNOFORTRAN=ON -DINTERFACE64=OFF
    -DBUILD_SINGLE=ON -DBUILD_DOUBLE=ON -DBUILD_COMPLEX=ON -DBUILD_COMPLEX16=ON
    -DBUILD_BFLOAT16=OFF -DUSE_THREAD=OFF -DUSE_OPENMP=OFF -DUSE_LOCKING=ON ${solver_blas_arch_args}
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel ${SOLVER_DEPENDENCY_JOBS}
  INSTALL_BYPRODUCTS "${solver_blas_install}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
    "${solver_blas_install}/include/openblas/cblas.h"
    "${solver_blas_install}/include/openblas/openblas_config.h"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON)
ExternalProject_Add_StepDependencies(solver_openblas_build patch
  "${openblas_patch_file}" "${solver_workspace_root}/dependencies.lock.json")
ExternalProject_Add_Step(solver_openblas_build licenses
  DEPENDEES install
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/NOTICE.txt"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${solver_blas_install}/share/solver"
  COMMAND "${CMAKE_COMMAND}" -E copy "<SOURCE_DIR>/LICENSE" "${solver_blas_install}/share/solver/OpenBLAS-LICENSE.txt"
  COMMAND "${CMAKE_COMMAND}" -E copy "<SOURCE_DIR>/lapack-netlib/LICENSE" "${solver_blas_install}/share/solver/LAPACK-LICENSE.txt"
  COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_LIST_DIR}/NOTICE.txt" "${solver_blas_install}/share/solver/OpenBLAS-NOTICE.txt")
