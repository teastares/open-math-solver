# 仅构建 HiGHS 锁定源码中的线性代数内核；与 CHOLMOD 共用主项目导入的完整 OpenBLAS。
set(solver_factor_source "${highs_SOURCE_DIR}/highs/ipm/hipo/factorhighs")
set(solver_factor_generated "${CMAKE_CURRENT_BINARY_DIR}/solver-factorhighs")
file(MAKE_DIRECTORY "${solver_factor_generated}")
set(solver_factor_sources)
foreach(unit Analyse CliqueStack DataCollector DenseFactHybrid
    FactorHiGHS FormatHandler HybridHybridFormatHandler HybridSolveHandler Numeric
    SolveHandler Swaps Symbolic)
  list(APPEND solver_factor_sources "${solver_factor_source}/${unit}.cpp")
endforeach()
# 配置时生成最小适配文件；不改下载源，不向 HiGHS 导出替换符号。
foreach(unit Factorise DgemmParallel CallAndTimeBlas DenseFactKernel)
  file(READ "${solver_factor_source}/${unit}.cpp" adapted_source)
  if(unit STREQUAL "CallAndTimeBlas")
    string(FIND "${adapted_source}" "#include \"HighsExternalApi.h\"" required_include)
    if(required_include LESS 0)
      message(FATAL_ERROR "锁定 BLAS 封装接口已变化，必须重新审查")
    endif()
    string(REPLACE "#include \"HighsExternalApi.h\"" "#include <cblas.h>" adapted_source "${adapted_source}")
    string(REPLACE "HighsExtras::blas::" "cblas_" adapted_source "${adapted_source}")
  elseif(unit STREQUAL "DenseFactKernel")
    # 上游同一主元会两次调用 staticReg；第二次的零增量不能覆盖已有扰动记录。
    string(FIND "${adapted_source}" "totalreg = pivot - old_pivot;" required_assignment)
    if(required_assignment LESS 0)
      message(FATAL_ERROR "锁定分解器正则记账接口已变化，必须重新审查")
    endif()
    string(REPLACE "totalreg = pivot - old_pivot;" "totalreg += pivot - old_pivot;"
      adapted_source "${adapted_source}")
  else()
    string(FIND "${adapted_source}" "#include \"parallel/HighsParallel.h\"" required_include)
    if(required_include LESS 0)
      message(FATAL_ERROR "锁定稀疏分解调度接口已变化，必须重新审查")
    endif()
    string(REPLACE "#include \"parallel/HighsParallel.h\"" "#include \"serial.hpp\"" adapted_source "${adapted_source}")
    string(REPLACE "highs::parallel::" "solver::factor_parallel::" adapted_source "${adapted_source}")
  endif()
  file(CONFIGURE OUTPUT "${solver_factor_generated}/${unit}.cpp" CONTENT "${adapted_source}" @ONLY)
  list(APPEND solver_factor_sources "${solver_factor_generated}/${unit}.cpp")
endforeach()
add_library(solver_factorhighs_kernel STATIC ${solver_factor_sources}
  "${highs_SOURCE_DIR}/highs/ipm/hipo/auxiliary/Auxiliary.cpp"
  "${highs_SOURCE_DIR}/highs/ipm/hipo/auxiliary/Logger.cpp")
target_include_directories(solver_factorhighs_kernel PRIVATE
  "${solver_factor_source}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/factorhighs")
target_link_libraries(solver_factorhighs_kernel PRIVATE highs::highs solver_openblas)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/factorhighs/NOTICE.txt"
  DESTINATION share/solver RENAME FactorHiGHS-NOTICE.txt)
install(FILES "${SOLVER_OPENBLAS_ROOT}/share/solver/OpenBLAS-LICENSE.txt"
  DESTINATION share/solver RENAME OpenBLAS-LICENSE.txt)

install(FILES "${SOLVER_OPENBLAS_ROOT}/share/solver/LAPACK-LICENSE.txt"
  DESTINATION share/solver RENAME LAPACK-LICENSE.txt)
