"""复用锁定的 v47 阶段计时核；只导出源码和静态字面复用检查。"""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PREFIX = "V99_PROFILE "
ENVIRONMENT = "SOLVER_V99_PROFILE"
SELFTEST_MACRO = "SOLVER_V99_PROFILE_SELFTEST"
HEADER_NAME = "v99-profile.hpp"
PLAN_SHA256 = "523115acd8348c1d21c7f3bc9ce302fedcd2a5b630ec65d96ee1235c8226de15"
PARENT_SHA256 = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
STAGES = (
    "total", "canonicalize", "kkt_setup", "normal_setup", "symbolic_analysis",
    "factor", "normal_factor", "rectangular_assembly", "explicit_gram_assembly",
    "cholmod_numeric", "factor_check", "hsd_constant_direct", "hsd_direction_direct",
    "direct", "normal_solve", "cholmod_solve", "preconditioner", "hsd_column_product",
    "hsd_true_product", "coupled", "point_update", "assess_homogeneous", "verify_local",
    "initialize", "run_homogeneous", "corrector_rhs", "homogeneous_step", "cone_multiply",
    "cone_transpose_multiply", "cholmod_factor_adapter", "solve_factored",
)


HEADER = r'''#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace solver_v99_profile {
struct Entry {
  std::int64_t inclusive_ns = 0, exclusive_ns = 0;
  std::uint64_t started = 0, completed = 0, unwound = 0;
};
inline constexpr const char* names[] = {@NAMES@};
inline thread_local std::array<Entry, @COUNT@> entries{};
inline thread_local bool enabled = false;
inline thread_local std::uint64_t clock_reads = 0;
struct Scope;
inline thread_local Scope* parent = nullptr;
#ifdef SOLVER_V99_PROFILE_SELFTEST
inline thread_local std::int64_t (*test_clock)() noexcept = nullptr;
#endif
inline std::int64_t now() noexcept {
  ++clock_reads;
#ifdef SOLVER_V99_PROFILE_SELFTEST
  if (test_clock) return test_clock();
#endif
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Scope {
  bool active = false;
  std::size_t index = 0;
  Scope* previous = nullptr;
  std::int64_t start = 0, children = 0;
  int exceptions = 0;
  explicit Scope(std::size_t value) noexcept {
    // 关闭时不取时间、不访问作用域链，也不分配内存。
    if (!enabled) return;
    active = true;
    index = value;
    previous = parent;
    exceptions = std::uncaught_exceptions();
    start = now();
    ++entries[index].started;
    parent = this;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept {
    if (!active) return;
    const auto duration = now() - start;
    auto& entry = entries[index];
    entry.inclusive_ns += duration;
    entry.exclusive_ns += duration - children;
    if (std::uncaught_exceptions() > exceptions) ++entry.unwound;
    else ++entry.completed;
    parent = previous;
    if (previous) previous->children += duration;
  }
};
template<class Function> decltype(auto) call(std::size_t index, Function&& function) {
  Scope scope(index);
  return function();
}
template<class Value, class Function> Value materialize(std::size_t index, Function&& function) {
  Scope scope(index);
  // Eigen 惰性表达式必须在计时作用域内物化，不能只计表达式构造。
  return Value(function());
}
inline std::int64_t exclusive_sum() noexcept {
  std::int64_t result = 0;
  for (const auto& entry : entries) result += entry.exclusive_ns;
  return result;
}
inline bool counts_closed() noexcept {
  std::uint64_t started = 0;
  for (const auto& entry : entries) {
    if (entry.started != entry.completed + entry.unwound) return false;
    started += entry.started;
  }
  return clock_reads == 2 * started;
}
struct Report {
  bool active = false;
  Report() noexcept {
    const auto* value = std::getenv("SOLVER_V99_PROFILE");
    enabled = value && value[0] == '1' && value[1] == '\0';
    if (!enabled) return;
    active = true;
    entries = {};
    clock_reads = 0;
    parent = nullptr;
  }
  ~Report() noexcept {
    if (!active) return;
    const auto sum = exclusive_sum();
    std::fputs("V99_PROFILE {\"schema\":1,\"stages\":{", stderr);
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto& entry = entries[i];
      std::fprintf(stderr, "%s\"%s\":{\"inclusive_ns\":%lld,\"exclusive_ns\":%lld,"
          "\"started\":%llu,\"completed\":%llu,\"unwound\":%llu}", i ? "," : "", names[i],
          static_cast<long long>(entry.inclusive_ns), static_cast<long long>(entry.exclusive_ns),
          static_cast<unsigned long long>(entry.started), static_cast<unsigned long long>(entry.completed),
          static_cast<unsigned long long>(entry.unwound));
    }
    std::fprintf(stderr, "},\"clock_reads\":%llu,\"exclusive_sum_ns\":%lld,"
        "\"total_inclusive_ns\":%lld,\"conserved\":%s,\"stack_empty\":%s,\"counts_closed\":%s}\n",
        static_cast<unsigned long long>(clock_reads), static_cast<long long>(sum),
        static_cast<long long>(entries[0].inclusive_ns), sum == entries[0].inclusive_ns ? "true" : "false",
        parent == nullptr ? "true" : "false", counts_closed() ? "true" : "false");
    enabled = false;
  }
};
}  // 命名空间 solver_v99_profile
'''.replace("@NAMES@", ",".join(json.dumps(name) for name in STAGES)).replace("@COUNT@", str(len(STAGES)))


SELF_MAIN = r'''#include "v99-profile.hpp"
void profile_cross_tu(bool fail);
namespace {
std::int64_t tick = 0;
std::int64_t fake_clock() noexcept { const auto result = tick; tick += 10; return result; }
}
int main() {
  using namespace solver_v99_profile;
  test_clock = fake_clock;
  Report report;
  const bool active = enabled;
  {
    Scope total(0);
    Scope outer(1);
    profile_cross_tu(false);
    try { profile_cross_tu(true); } catch (int value) { if (value != 7) return 3; }
  }
  bool passed = parent == nullptr && counts_closed();
  if (active) {
    passed = passed && clock_reads == 12 && entries[0].inclusive_ns == 110 &&
        entries[0].exclusive_ns == 20 && entries[1].inclusive_ns == 110 &&
        entries[1].exclusive_ns == 50 && entries[3].inclusive_ns == 60 &&
        entries[3].exclusive_ns == 40 && entries[1].started == 3 &&
        entries[1].completed == 2 && entries[1].unwound == 1 &&
        entries[3].started == 2 && entries[3].completed == 1 && entries[3].unwound == 1 &&
        exclusive_sum() == entries[0].inclusive_ns;
  } else {
    passed = passed && clock_reads == 0 && tick == 0;
    for (const auto& entry : entries) passed = passed && entry.started == 0 && entry.inclusive_ns == 0;
  }
  std::printf("{\"passed\":%s,\"enabled\":%s}\n", passed ? "true" : "false", active ? "true" : "false");
  return passed ? 0 : 2;
}
'''
SELF_CROSS = r'''#include "v99-profile.hpp"
void profile_cross_tu(bool fail) {
  solver_v99_profile::Scope outer(3);
  solver_v99_profile::Scope inner(1);
  if (fail) throw 7;
}
'''


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _bindings(value):
    """固定替换实验身份，不更改旧计时核的数据流或双翻译单元自测。"""
    return (value.replace("solver_v47_profile", "solver_v99_profile")
            .replace("SOLVER_V47_PROFILE", "SOLVER_V99_PROFILE")
            .replace("V47_PROFILE", "V99_PROFILE")
            .replace("v47-profile.hpp", "v99-profile.hpp"))


def _parent_literals(source):
    """只读取顶层字面量；不导入、不执行旧构建器。"""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
    header = values["HEADER"]
    # 锁定的父 HEADER 只有两次占位符替换，其底层是原始字符串字面量。
    for placeholder in ("@COUNT@", "@NAMES@"):
        _require(isinstance(header, ast.Call) and isinstance(header.func, ast.Attribute)
                 and header.func.attr == "replace" and len(header.args) == 2
                 and ast.literal_eval(header.args[0]) == placeholder,
                 "锁定父 HEADER 的占位符结构改变")
        header = header.func.value
    template = ast.literal_eval(header)
    _require(isinstance(template, str), "锁定父 HEADER 不是字符串字面量")
    return template, ast.literal_eval(values["SELF_MAIN"]), ast.literal_eval(values["SELF_CROSS"])


def static_check(root=None):
    """核对冻结身份、阶段顺序及允许身份替换后的三个原始源码块。"""
    repository = ROOT if root is None else Path(root)
    data = repository / "knowledge/experiments/data/2026-10-12-halpern-ipm"
    plan_path = data / "v99-plan.json"
    plan_bytes = plan_path.read_bytes()
    _require(_digest(plan_bytes) == PLAN_SHA256, "v99 冻结计划身份不符")
    plan = json.loads(plan_bytes)
    _require(tuple(plan["stages"]) == STAGES and len(STAGES) == len(set(STAGES)) == 31
             and STAGES[0] == "total", "v99 阶段顺序或数量不符")
    _require(plan["profile_environment"] == ENVIRONMENT and plan["profile_marker"] + " " == PREFIX,
             "v99 计时环境或输出前缀不符")
    _require(plan["parent_profile_helper"] == "v47-profile-build.py"
             and plan["parent_profile_helper_sha256"] == PARENT_SHA256,
             "v99 父计时核锁不符")
    parent_path = data / plan["parent_profile_helper"]
    parent_bytes = parent_path.read_bytes()
    _require(_digest(parent_bytes) == PARENT_SHA256, "锁定 v47 计时核源码身份不符")
    template, parent_main, parent_cross = _parent_literals(parent_bytes.decode("utf-8-sig"))
    # AST 的字符串值统一换行，和 Python 正常读取源码时的字符串值一致。
    expected_header = (_bindings(template).replace("@NAMES@", ",".join(json.dumps(name) for name in STAGES))
                       .replace("@COUNT@", str(len(STAGES))))
    _require(HEADER == expected_header, "v99 HEADER 超出允许的身份或阶段替换")
    _require(SELF_MAIN == _bindings(parent_main), "v99 双翻译单元主自测超出允许的身份替换")
    _require(SELF_CROSS == _bindings(parent_cross), "v99 双翻译单元交叉自测超出允许的身份替换")
    return {
        "passed": True,
        "stages": list(STAGES),
        "plan_sha256": PLAN_SHA256,
        "parent_profile_helper_sha256": PARENT_SHA256,
        "header_sha256": _digest(HEADER.encode("utf-8")),
        "self_main_sha256": _digest(SELF_MAIN.encode("utf-8")),
        "self_cross_sha256": _digest(SELF_CROSS.encode("utf-8")),
        "parent_header_reused": True,
        "parent_self_main_reused": True,
        "parent_self_cross_reused": True,
        "scope": "仅静态身份与字面复用检查；不编译、不运行 LP",
    }
