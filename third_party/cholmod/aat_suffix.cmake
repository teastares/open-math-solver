# CHOLMOD 超节点数值内核归 Timothy A. Davis，保留上游 GPL-2.0-or-later 许可。
# 仅从依赖锁提供的源码生成私有适配副本；不修改下载目录，也不替换上游符号。
# 生产调用只使用已验证的 fused AAT 路径：转置结构及原 CSC 槽固定，数值读当前 A。
include_guard(GLOBAL)

# 精确且唯一的文本锚点；上游发生变化时停止配置，禁止悄悄漏用或重复应用补丁。
function(solver_cholmod_replace_once variable before after)
  string(FIND "${${variable}}" "${before}" first)
  if(first LESS 0)
    message(FATAL_ERROR "锁定 CHOLMOD 适配锚点缺失，必须重新审查：${variable}")
  endif()
  string(LENGTH "${before}" width)
  math(EXPR tail_start "${first} + ${width}")
  string(SUBSTRING "${${variable}}" ${tail_start} -1 tail)
  string(FIND "${tail}" "${before}" duplicate)
  if(NOT duplicate LESS 0)
    message(FATAL_ERROR "锁定 CHOLMOD 适配锚点不唯一，必须重新审查：${variable}")
  endif()
  string(REPLACE "${before}" "${after}" changed "${${variable}}")
  set(${variable} "${changed}" PARENT_SCOPE)
endfunction()

function(solver_add_cholmod_aat_suffix)
  if(NOT TARGET CHOLMOD_static OR NOT TARGET SuiteSparse::CHOLMOD_static)
    message(FATAL_ERROR "私有 AAT 内核必须在锁定 CHOLMOD 静态目标创建后配置")
  endif()
  if(NOT CHOLMOD_GPL OR NOT CHOLMOD_SUPERNODAL OR
      CHOLMOD_USE_OPENMP OR CHOLMOD_USE_CUDA)
    message(FATAL_ERROR "私有 AAT 内核要求现有 GPL 超节点、单线程且无 CUDA 的构建配置")
  endif()
  set(upstream "${suitesparse_SOURCE_DIR}/CHOLMOD/Supernodal")
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/solver-cholmod-aat")
  file(MAKE_DIRECTORY "${generated}")
  foreach(unit cholmod_l_super_numeric.c cholmod_super_numeric.c t_cholmod_super_numeric_worker.c)
    if(NOT EXISTS "${upstream}/${unit}")
      message(FATAL_ERROR "锁定 CHOLMOD 数值源码缺失：${upstream}/${unit}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${upstream}/${unit}")
  endforeach()

  file(READ "${upstream}/cholmod_super_numeric.c" numeric)
  file(READ "${upstream}/t_cholmod_super_numeric_worker.c" worker)
  # 统一输入换行后匹配；不改变任何浮点表达式、遍历顺序或模板类型。
  string(REPLACE "\r\n" "\n" numeric "${numeric}")
  string(REPLACE "\r\n" "\n" worker "${worker}")
  solver_cholmod_replace_once(numeric
    "int CHOLMOD(super_numeric)\n("
    "int solver_cholmod_aat_numeric\n(")
  set(signature_before [=[    cholmod_common *Common
)
{]=])
  set(signature_after [=[    cholmod_common *Common,
    const uint32_t *source_slots
)
{]=])
  solver_cholmod_replace_once(numeric "${signature_before}" "${signature_after}")
  solver_cholmod_replace_once(worker "${signature_before}" "${signature_after}")

  # F 仅持有固定转置结构；fused 模式不分配或刷新 F 的浮点数值。
  # 原有类型、形状检查继续执行；槽映射由唯一生产适配器构造并保持同一结构世代。
  set(transpose_before [=[        RETURN_IF_XTYPE_INVALID (F, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;]=])
  set(transpose_after [=[        if (source_slots == NULL || F->x != NULL)
        {
            RETURN_IF_XTYPE_INVALID (F, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE) ;
        }
        if (source_slots != NULL &&
            (!A->packed || !A->sorted || !F->packed || !F->sorted ||
                A->xtype != CHOLMOD_REAL || A->dtype != CHOLMOD_DOUBLE))
        {
            ERROR (CHOLMOD_INVALID, "suffix assembly requires the fixed packed real transpose") ;
            return (FALSE) ;
        }]=])
  solver_cholmod_replace_once(numeric "${transpose_before}" "${transpose_after}")

  # 只有实数 double 模板消费槽映射，其余上游模板保持原调用行为。
  foreach(kind rs cs zs)
    solver_cholmod_replace_once(numeric
      "${kind}_cholmod_super_numeric_worker (A, F, s_beta, L, C, Common)"
      "${kind}_cholmod_super_numeric_worker (A, F, s_beta, L, C, Common, NULL)")
  endforeach()
  solver_cholmod_replace_once(numeric
    "rd_cholmod_super_numeric_worker (A, F, beta, L, C, Common)"
    "rd_cholmod_super_numeric_worker (A, F, beta, L, C, Common, source_slots)")
  foreach(kind cd zd)
    solver_cholmod_replace_once(numeric
      "${kind}_cholmod_super_numeric_worker (A, F, beta, L, C, Common)"
      "${kind}_cholmod_super_numeric_worker (A, F, beta, L, C, Common, NULL)")
  endforeach()

  set(assembly_before [=[                    // fjk = Fx [pf]
                    L_ASSIGN (fjk,0, Fx,Fz,pf) ;

                    Int p = Ap [j] ;
                    Int pend = (Apacked) ? (Ap [j+1]) : (p + Anz [j]) ;]=])
  set(assembly_after [=[                    // 固定转置槽定位本项在原 CSC 列中的位置；数值直接读当前 A。
                    Int p = source_slots ? (Int) source_slots [pf] : Ap [j] ;
                    #if defined (REAL) && defined (DOUBLE)
                    if (Fx == NULL)
                    {
                        fjk [0] = Ax [p] ;
                    }
                    else
                    #endif
                    {
                        L_ASSIGN (fjk,0, Fx,Fz,pf) ;
                    }
                    Int pend = (Apacked) ? (Ap [j+1]) : (p + Anz [j]) ;]=])
  solver_cholmod_replace_once(worker "${assembly_before}" "${assembly_after}")
  set(suffix_before [=[                        if (i >= k)
                        {
                            // See the discussion of imap above.]=])
  set(suffix_after [=[                        if (source_slots || i >= k)
                        {
                            // See the discussion of imap above.]=])
  solver_cholmod_replace_once(worker "${suffix_before}" "${suffix_after}")

  # 私有双精度实数 AAT 调用提供压缩的 A、F 及原 CSC 槽映射。
  # 用一次等价的无符号范围检查，并保持原乘加顺序。
  set(packed_before [=[            else
            {

                //--------------------------------------------------------------
                // copy the kth column of A*F into the supernode]=])
  set(packed_after [=[            #if defined (REAL) && defined (DOUBLE)
            else if (source_slots && Fx == NULL && Apacked && Fpacked)
            {
                const Int panel_col = psx + (k-k1)*nsrow;
                for (Int pf = Fp[k]; pf < Fp[k+1]; ++pf)
                {
                    const Int start = (Int) source_slots[pf];
                    const Int end = Ap[Fi[pf]+1];
                    const double fjk = Ax[start];
                    for (Int p = start; p < end; ++p)
                    {
                        const Int imap = Map[Ai[p]];
                        if ((uint64_t) imap < (uint64_t) nsrow)
                            Lx[panel_col+imap] += Ax[p] * fjk;
                    }
                }
            }
            #endif
            else
            {

                //--------------------------------------------------------------
                // copy the kth column of A*F into the supernode]=])
  solver_cholmod_replace_once(worker "${packed_before}" "${packed_after}")

  # 三层双引号包含必须都指向构建目录，否则编译器会优先采用下载目录中的原 worker。
  configure_file("${upstream}/cholmod_l_super_numeric.c"
    "${generated}/cholmod_l_super_numeric.c" COPYONLY)
  file(CONFIGURE OUTPUT "${generated}/cholmod_super_numeric.c" CONTENT "${numeric}" @ONLY NEWLINE_STYLE LF)
  file(CONFIGURE OUTPUT "${generated}/t_cholmod_super_numeric_worker.c" CONTENT "${worker}" @ONLY NEWLINE_STYLE LF)

  add_library(solver_cholmod_aat_suffix STATIC "${generated}/cholmod_l_super_numeric.c")
  set_target_properties(solver_cholmod_aat_suffix PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS ON)
  # 继承原目标实际编译配置，避免手写另一套 BLAS、平台和模块宏。
  target_compile_definitions(solver_cholmod_aat_suffix PRIVATE
    "$<TARGET_PROPERTY:CHOLMOD_static,COMPILE_DEFINITIONS>")
  target_compile_options(solver_cholmod_aat_suffix PRIVATE
    "$<TARGET_PROPERTY:CHOLMOD_static,COMPILE_OPTIONS>")
  target_include_directories(solver_cholmod_aat_suffix PRIVATE
    "$<TARGET_PROPERTY:CHOLMOD_static,INCLUDE_DIRECTORIES>")
  target_link_libraries(solver_cholmod_aat_suffix PRIVATE SuiteSparse::CHOLMOD_static)
endfunction()

solver_add_cholmod_aat_suffix()
