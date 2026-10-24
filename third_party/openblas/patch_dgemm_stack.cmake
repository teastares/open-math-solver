cmake_minimum_required(VERSION 3.26)

# 本地修复仅作用于依赖锁中的 OpenBLAS 0.3.33 源文件；未知内容直接拒绝。
if(NOT DEFINED SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
  message(FATAL_ERROR "需要指定已解包的 OpenBLAS SOURCE_DIR")
endif()
get_filename_component(solver_workspace_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
file(READ "${solver_workspace_root}/dependencies.lock.json" dependency_lock)
string(JSON openblas_version GET "${dependency_lock}" openblas version)
if(NOT openblas_version STREQUAL "0.3.33")
  message(FATAL_ERROR "本补丁尚未审核当前 OpenBLAS 版本")
endif()
string(JSON patch_path GET "${dependency_lock}" openblas local_patches 0 path)
string(JSON patch_hash GET "${dependency_lock}" openblas local_patches 0 sha256)
string(JSON source_path GET "${dependency_lock}" openblas local_patches 0 source_file)
string(JSON source_hash GET "${dependency_lock}" openblas local_patches 0 source_sha256)
string(JSON patched_hash GET "${dependency_lock}" openblas local_patches 0 patched_sha256)
file(SHA256 "${CMAKE_CURRENT_LIST_FILE}" actual_patch_hash)
if(NOT patch_path STREQUAL "third_party/openblas/patch_dgemm_stack.cmake"
   OR NOT actual_patch_hash STREQUAL patch_hash)
  message(FATAL_ERROR "OpenBLAS 补丁脚本与依赖锁不一致")
endif()
if(DEFINED EXPECTED_PATCH_SHA256 AND NOT EXPECTED_PATCH_SHA256 STREQUAL patch_hash)
  message(FATAL_ERROR "OpenBLAS 构建命令的补丁校验值已过期，请重新配置")
endif()
if(NOT source_path STREQUAL "kernel/x86_64/dgemm_kernel_16x2_skylakex.S")
  message(FATAL_ERROR "OpenBLAS 补丁目标不在已审核范围内")
endif()
set(source_file "${SOURCE_DIR}/${source_path}")
file(SHA256 "${source_file}" actual_source_hash)
if(actual_source_hash STREQUAL patched_hash)
  message(STATUS "OpenBLAS DGEMM 栈容量补丁已应用，无需重复写入")
  return()
endif()
if(NOT actual_source_hash STREQUAL source_hash)
  message(FATAL_ERROR "OpenBLAS DGEMM 原始源码校验失败，拒绝修改")
endif()
file(READ "${source_file}" source_text)

# 完整锚点只能出现一次，防止新版本或手工改动被静默误替换。
function(replace_once old_text new_text)
  string(FIND "${source_text}" "${old_text}" first_position)
  if(first_position LESS 0)
    message(FATAL_ERROR "OpenBLAS 补丁找不到已审核的唯一锚点")
  endif()
  string(LENGTH "${old_text}" old_length)
  math(EXPR after_position "${first_position} + ${old_length}")
  string(SUBSTRING "${source_text}" ${after_position} -1 remaining_text)
  string(FIND "${remaining_text}" "${old_text}" duplicate_position)
  if(NOT duplicate_position LESS 0)
    message(FATAL_ERROR "OpenBLAS 补丁锚点重复，拒绝修改")
  endif()
  string(REPLACE "${old_text}" "${new_text}" result_text "${source_text}")
  set(source_text "${result_text}" PARENT_SCOPE)
endfunction()

replace_once([=[#define L_BUFFER_SIZE 512*8*4
#define LB2_OFFSET    512*8*2]=] [=[/* 两个缓冲区各存放三列；按本内核最大 K 分配并保持缓存行对齐。 */
#define LB2_OFFSET (((3 * DGEMM_DEFAULT_Q * SIZE + 63) / 64) * 64)
#define L_BUFFER_SIZE (2 * LB2_OFFSET)]=])

replace_once([=[#if defined(OS_WINDOWS)
#if   L_BUFFER_SIZE > 16384
#define STACK_TOUCH \
        movl    $0,  4096 * 4(%rsp);\
        movl    $0,  4096 * 3(%rsp);\
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 12288
#define STACK_TOUCH \
        movl    $0,  4096 * 3(%rsp);\
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 8192
#define STACK_TOUCH \
        movl    $0,  4096 * 2(%rsp);\
        movl    $0,  4096 * 1(%rsp);
#elif L_BUFFER_SIZE > 4096
#define STACK_TOUCH \
        movl    $0,  4096 * 1(%rsp);
#else
#define STACK_TOUCH
#endif
#else
#define STACK_TOUCH
#endif]=] [=[#if defined(OS_WINDOWS)
/* 从已触碰的保存区逐页下降，同时覆盖向下对齐增加的栈空间。 */
#define STACK_TOUCH \
        movq    SP, %rax;\
1:      subq    $4096, %rax;\
        cmpq    %rsp, %rax;\
        jbe     2f;\
        movl    $0, (%rax);\
        jmp     1b;\
2:      movl    $0, (%rsp);
#else
#define STACK_TOUCH
#endif]=])

string(SHA256 result_hash "${source_text}")
if(NOT result_hash STREQUAL patched_hash)
  message(FATAL_ERROR "OpenBLAS 补丁结果与依赖锁不一致，未写入源码")
endif()
# 固定 LF，避免 Windows 的文本写入转换改变锁定校验值。
file(CONFIGURE OUTPUT "${source_file}" CONTENT "${source_text}" @ONLY NEWLINE_STYLE UNIX)
file(SHA256 "${source_file}" written_hash)
if(NOT written_hash STREQUAL patched_hash)
  message(FATAL_ERROR "OpenBLAS 补丁写入校验失败")
endif()
message(STATUS "已应用 OpenBLAS DGEMM 栈容量与逐页触碰修复")
