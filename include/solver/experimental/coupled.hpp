#pragma once

#include "solver/experimental/fma_axpy.hpp"
#include "solver/experimental/profile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace solver::v100_coupled {

using Index = std::int64_t;
using Budget = std::function<void()>;
using Preconditioner = std::function<std::vector<double>(std::span<const double>, bool refinement)>;
using ProductDouble = std::function<void(std::span<const double>, std::vector<long double>&)>;
using ProductWide = std::function<void(std::span<const long double>, std::vector<long double>&)>;
static_assert(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits,
    "完整物理宽方向要求long double的有效位多于double");

inline constexpr std::size_t max_window = 32;
inline constexpr int max_cycles = 3;
inline constexpr std::size_t budget_batch = 4096;

struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

struct InitialForcing {
  std::array<double, 3> rhs_norms{};
};

struct InitialForcingWork {
  std::array<double, 3> rhs_norms{};
  std::optional<std::array<long double, 5>> defect_norms;
  std::optional<long double> measure;
  const char* decision = "not_checked";
};

struct CandidateForcingWork {
  Index cycle = 0, krylov_step = 0;
  std::optional<std::array<long double, 5>> defect_norms;
  std::optional<long double> strict_measure, best_measure_before, measure;
  const char* decision = "not_checked";
};

// 这里只规定方程行的五块顺序；物理未知量的排列由预条件器与算子共同规定。
struct Layout {
  std::size_t columns = 0, rows = 0, inequalities = 0;
  std::size_t dimension() const {
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (inequalities > rows || columns > limit - rows ||
        columns + rows > limit - inequalities || columns + rows + inequalities > limit - 2)
      throw std::invalid_argument("联立Newton五块维数无效或溢出");
    return columns + rows + inequalities + 2;
  }
  std::size_t block(std::size_t index) const noexcept {
    const auto gap = columns + rows;
    return index < columns ? 0 : index < gap ? 1 : index == gap ? 2 :
        index < gap + 1 + inequalities ? 3 : 4;
  }
};

// 只记已启动/返回的实际调用和已完成的真验快照，不保存方向或模型引用。
struct Work {
  std::size_t dimension = 0, window = 0;
  Index preconditioner_started = 0, preconditioner_completed = 0;
  Index products_started = 0, products_completed = 0;
  Index krylov_started = 0, krylov_completed = 0, cycles_started = 0;
  Index true_checks_started = 0, true_checks_completed = 0;
  const char* status = "not_started";
  std::optional<long double> initial_measure, final_measure;
  std::array<long double, 5> block_measures{};
  std::array<double, 5> tolerances{};
  Index initial_forcing_checks = 0, initial_forcing_returns = 0;
  std::optional<InitialForcingWork> initial_forcing;
  Index candidate_forcing_checks = 0, candidate_forcing_returns = 0;
  std::optional<CandidateForcingWork> candidate_forcing;
};

namespace implementation {

inline void budget(const Budget& check) { if (check) check(); }
inline void coordinate_budget(std::size_t index, const Budget& check) {
  if ((index & (budget_batch - 1)) == 0) budget(check);
}
[[noreturn]] inline void arithmetic(Work& work, const char* message) {
  work.status = "numerical_error";
  throw Failure(message);
}
inline double narrow(long double value, Work& work) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result))
    arithmetic(work, "联立Newton搜索算术非有限");
  return result;
}
inline void finite(std::span<const double> values, Work& work, const Budget& check) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    if (!std::isfinite(values[i])) arithmetic(work, "联立Newton物理方向非有限");
  }
  budget(check);
}
inline void finite(std::span<const long double> values, Work& work, const Budget& check) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    if (!std::isfinite(values[i])) arithmetic(work, "联立Newton宽物理方向非有限");
  }
  budget(check);
}
inline double norm(std::span<const double> values, Work& work, const Budget& check) {
  solver_v112_profile::Scope v112_norm_scope(solver_v112_profile::Stage::norm);
  long double scale = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    if (!std::isfinite(values[i])) arithmetic(work, "联立Newton搜索向量非有限");
    scale = std::max(scale, std::abs(static_cast<long double>(values[i])));
  }
  if (scale == 0) { budget(check); return 0; }
  long double squares = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    coordinate_budget(i, check);
    const long double ratio = static_cast<long double>(values[i]) / scale;
    squares += ratio * ratio;
  }
  budget(check);
  return narrow(scale * std::sqrt(squares), work);
}
inline double dot(std::span<const double> left, std::span<const double> right,
    Work& work, const Budget& check) {
  solver_v112_profile::Scope v112_dot_scope(solver_v112_profile::Stage::dot);
  long double sum = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    coordinate_budget(i, check);
    sum += static_cast<long double>(left[i]) * right[i];
  }
  budget(check);
  return narrow(sum, work);
}
struct ExceptionStatus {
  Work& work;
  ~ExceptionStatus() noexcept {
    if (std::string_view(work.status) == "running") work.status = "exception";
  }
};
struct Assessment {
  std::vector<long double> residual;
  std::array<long double, 5> blocks{};
  long double measure = 0;
};

}  // 命名空间 implementation

// 初P及显式启用的FG候选先验严格门槛；forcing只对三个线性块使用同一受控界。
// T默认取strict五块门槛；显式forcing_weights时只将搜索尺度换为同一F。
// 真实strict/forcing验收仍用原门槛；Arnoldi作用于T^-1 A P T。
// 每个周期的基点固定，重启保留全部完整真验中max-block-ratio最小的物理LD点。
inline std::vector<long double> solve(const Layout& layout, std::span<const double> rhs,
    const std::array<double, 5>& tolerances, const Preconditioner& precondition,
    const ProductDouble& column_multiply, const ProductWide& true_multiply,
    Work& work, const Budget& check = {}, const InitialForcing* initial = nullptr, bool allow_candidates = false,
    bool forcing_weights = false, bool hardware_axpy = false) {
  solver_v112_profile::Scope v112_coupled_scope(solver_v112_profile::Stage::coupled);
  work = Work{};
  work.status = "running";
  implementation::ExceptionStatus exception_status{work};
  const auto dimension = layout.dimension();
  work.dimension = dimension;
  work.window = std::min(max_window, dimension);
  work.tolerances = tolerances;
  if (initial) {
    work.initial_forcing.emplace();
    work.initial_forcing->rhs_norms = initial->rhs_norms;
  }
  if (allow_candidates) work.candidate_forcing.emplace();
  implementation::budget(check);
  if (forcing_weights && (!allow_candidates || !initial))
    throw std::invalid_argument("受控FG搜索尺度要求候选forcing与原完整右端块范数");
  if (allow_candidates && !initial)
    throw std::invalid_argument("候选forcing必须提供原完整右端块范数");
  if (rhs.size() != dimension || !precondition || !column_multiply || !true_multiply)
    throw std::invalid_argument("联立Newton右端、回调或维数无效");
  for (double value : tolerances)
    if (!(value > 0) || !std::isfinite(value))
      throw std::invalid_argument("联立Newton五块门槛必须有限正");
  if (work.initial_forcing)
    for (double value : work.initial_forcing->rhs_norms)
      if (!(value >= 0) || !std::isfinite(value))
        throw std::invalid_argument("初P forcing右端块范数必须有限非负");
  implementation::finite(rhs, work, check);
  const auto apply = [&](std::span<const double> value, bool refinement) {
    implementation::budget(check);
    ++work.preconditioner_started;
    auto result = precondition(value, refinement);
    ++work.preconditioner_completed;
    implementation::budget(check);
    if (result.size() != dimension) throw std::invalid_argument("联立Newton预条件方向维数不一致");
    implementation::finite(result, work, check);
    return result;
  };
  const auto product = [&]<typename Scalar>(std::span<const Scalar> value, std::vector<long double>& image) {
    implementation::budget(check);
    ++work.products_started;
    if constexpr (std::is_same_v<Scalar, double>) column_multiply(value, image);
    else true_multiply(value, image);
    ++work.products_completed;
    implementation::budget(check);
    if (image.size() != dimension) throw std::invalid_argument("联立Newton五块乘积维数不一致");
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      if (!std::isfinite(image[i])) implementation::arithmetic(work, "联立Newton真实乘积非有限");
    }
    implementation::budget(check);
  };
  const auto assess = [&](std::span<const long double> direction, std::array<long double, 5>* defects = nullptr) {
    implementation::budget(check);
    ++work.true_checks_started;
    implementation::Assessment result;
    product(direction, result.residual);
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      result.residual[i] = static_cast<long double>(rhs[i]) - result.residual[i];
      const auto block = layout.block(i);
      const long double ratio = std::abs(result.residual[i]) / tolerances[block];
      if (!std::isfinite(ratio)) implementation::arithmetic(work, "联立Newton真实残差非有限");
      if (defects) (*defects)[block] = std::max((*defects)[block], std::abs(result.residual[i]));
      result.blocks[block] = std::max(result.blocks[block], ratio);
      result.measure = std::max(result.measure, ratio);
    }
    implementation::budget(check);
    ++work.true_checks_completed;
    return result;
  };
  // 只有初P结果做一次预算化提升；Arnoldi列Z保持double，不额外复制大向量。
  auto best = [&] {
    const auto initial = apply(rhs, false);
    std::vector<long double> promoted(dimension);
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      promoted[i] = initial[i];
    }
    implementation::budget(check);
    return promoted;
  }();
  std::array<long double, 5> initial_defects{};
  auto best_assessment = assess(best, work.initial_forcing ? &initial_defects : nullptr);
  work.initial_measure = best_assessment.measure;
  work.final_measure = best_assessment.measure;
  work.block_measures = best_assessment.blocks;
  // 初始真验已完整返回才发布E5，任何中途预算均不能暴露半份初始快照。
  if (work.initial_forcing) work.initial_forcing->defect_norms = initial_defects;
  const auto accept = [&] {
    // 真验已过也仍需最后预算；异常不能把status提前记作严格返回。
    implementation::budget(check);
    work.status = "accepted";
  };
  if (best_assessment.measure <= 1) {
    if (work.initial_forcing) work.initial_forcing->decision = "strict_initial";
    accept(); return best;
  }
  if (work.initial_forcing) {
    auto& proof = *work.initial_forcing;
    long double measure = 0;
    for (std::size_t block = 0; block < 5; ++block) {
      long double limit = tolerances[block];
      if (block < 3) limit = std::max(limit, 0.1L * static_cast<long double>(proof.rhs_norms[block]));
      measure = std::max(measure, initial_defects[block] / limit);
    }
    proof.measure = measure;
    proof.decision = measure <= 1 ? "eligible" : "rejected";
    ++work.initial_forcing_checks;
    if (measure <= 1) {
      implementation::budget(check);
      proof.decision = "returned";
      work.status = "forced_initial";
      ++work.initial_forcing_returns;
      return best;
    }
  }
  std::array<long double, 5> search_scales{};
  for (std::size_t block = 0; block < 5; ++block) {
    search_scales[block] = tolerances[block];
    if (forcing_weights && block < 3)
      search_scales[block] = std::max(search_scales[block],
          0.1L * static_cast<long double>(work.initial_forcing->rhs_norms[block]));
  }
  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    implementation::budget(check);
    ++work.cycles_started;
    const auto base = best;
    std::vector<std::vector<double>> basis(1, std::vector<double>(dimension));
    std::vector<std::vector<double>> directions;
    directions.reserve(work.window);
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      basis[0][i] = implementation::narrow(best_assessment.residual[i] / search_scales[layout.block(i)], work);
    }
    const double beta = implementation::norm(basis[0], work, check);
    if (!(beta > 0)) implementation::arithmetic(work, "联立Newton初始Krylov残差无效");
    for (std::size_t i = 0; i < dimension; ++i) {
      implementation::coordinate_budget(i, check);
      basis[0][i] /= beta;
    }
    std::vector<std::vector<double>> upper(work.window, std::vector<double>(work.window + 1));
    std::vector<double> cosine(work.window), sine(work.window), transformed(work.window + 1);
    transformed[0] = beta;
    for (std::size_t column = 0; column < work.window; ++column) {
      implementation::budget(check);
      ++work.krylov_started;
      auto physical_rhs = basis[column];
      for (std::size_t i = 0; i < dimension; ++i) {
        implementation::coordinate_budget(i, check);
        physical_rhs[i] = implementation::narrow(static_cast<long double>(physical_rhs[i]) * search_scales[layout.block(i)], work);
      }
      directions.push_back(apply(physical_rhs, true));
      std::vector<long double> image;
      product(std::span<const double>(directions.back()), image);
      std::vector<double> value(dimension);
      for (std::size_t i = 0; i < dimension; ++i) {
        implementation::coordinate_budget(i, check);
        value[i] = implementation::narrow(image[i] / search_scales[layout.block(i)], work);
      }
      // 两遍改进Gram-Schmidt保持原double搜索；物理候选的系数乘积和交付始终LD。
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t row = 0; row <= column; ++row) {
          const double projection = implementation::dot(basis[row], value, work, check);
          upper[column][row] = implementation::narrow(static_cast<long double>(upper[column][row]) + projection, work);
          solver_v112_profile::Scope v112_axpy_scope(solver_v112_profile::Stage::axpy);
          if (hardware_axpy) {
            const auto updated = ::solver::experimental::fma_axpy::subtract_scaled(value, basis[row], projection, check, true);
            if (!updated.finite) implementation::arithmetic(work, "联立Newton搜索算术非有限");
          } else {
            for (std::size_t i = 0; i < dimension; ++i) {
              implementation::coordinate_budget(i, check);
              value[i] = implementation::narrow(static_cast<long double>(value[i]) -
                  static_cast<long double>(projection) * basis[row][i], work);
            }
          }
        }
      }
      const double remaining = implementation::norm(value, work, check);
      upper[column][column + 1] = remaining;
      if (remaining > 0) {
        for (std::size_t i = 0; i < dimension; ++i) {
          implementation::coordinate_budget(i, check);
          value[i] /= remaining;
        }
        basis.push_back(std::move(value));
      }
      for (std::size_t row = 0; row < column; ++row) {
        implementation::budget(check);
        const double first = upper[column][row], second = upper[column][row + 1];
        upper[column][row] = implementation::narrow(static_cast<long double>(cosine[row]) * first +
            static_cast<long double>(sine[row]) * second, work);
        upper[column][row + 1] = implementation::narrow(-static_cast<long double>(sine[row]) * first +
            static_cast<long double>(cosine[row]) * second, work);
      }
      const double diagonal = std::hypot(upper[column][column], upper[column][column + 1]);
      if (!(diagonal > 0) || !std::isfinite(diagonal)) {
        work.status = "breakdown";
        throw Failure("联立NewtonKrylov子空间退化");
      }
      cosine[column] = upper[column][column] / diagonal;
      sine[column] = upper[column][column + 1] / diagonal;
      upper[column][column] = diagonal;
      upper[column][column + 1] = 0;
      transformed[column + 1] = implementation::narrow(-static_cast<long double>(sine[column]) * transformed[column], work);
      transformed[column] = implementation::narrow(static_cast<long double>(transformed[column]) * cosine[column], work);
      implementation::budget(check);
      ++work.krylov_completed;
      if (std::abs(transformed[column + 1]) <= 1 || column + 1 == work.window || remaining == 0) {
        std::vector<double> coefficients(transformed.begin(), transformed.begin() + static_cast<std::ptrdiff_t>(column + 1));
        for (std::size_t row = column + 1; row-- > 0;) {
          implementation::budget(check);
          long double value = coefficients[row];
          for (std::size_t j = row + 1; j <= column; ++j)
            value -= static_cast<long double>(upper[j][row]) * coefficients[j];
          coefficients[row] = implementation::narrow(value / upper[row][row], work);
        }
        auto candidate = base;
        for (std::size_t i = 0; i < dimension; ++i) {
          implementation::coordinate_budget(i, check);
          long double value = base[i];
          for (std::size_t j = 0; j <= column; ++j)
            value += static_cast<long double>(coefficients[j]) * directions[j][i];
          if (!std::isfinite(value)) implementation::arithmetic(work, "联立Newton宽物理候选非有限");
          candidate[i] = value;
        }
        std::array<long double, 5> candidate_defects{};
        auto checked = assess(candidate, allow_candidates ? &candidate_defects : nullptr);
        const long double best_measure_before = best_assessment.measure;
        const bool candidate_improved = checked.measure < best_assessment.measure;
        if (candidate_improved) {
          best.swap(candidate);
          best_assessment = std::move(checked);
          work.final_measure = best_assessment.measure;
          work.block_measures = best_assessment.blocks;
        }
        if (best_assessment.measure <= 1) { accept(); return best; }
        if (allow_candidates) {
          // swap后的引用仍须绑定本次已验候选，不能把未改善候选误换为旧best。
          auto& actual_candidate = candidate_improved ? best : candidate;
          const auto& actual_assessment = candidate_improved ? best_assessment : checked;
          CandidateForcingWork proof;
          proof.cycle = work.cycles_started;
          proof.krylov_step = work.krylov_completed;
          proof.defect_norms = candidate_defects;
          proof.strict_measure = actual_assessment.measure;
          proof.best_measure_before = best_measure_before;
          long double measure = 0;
          for (std::size_t block = 0; block < 5; ++block) {
            long double limit = tolerances[block];
            if (block < 3) limit = std::max(limit, 0.1L * static_cast<long double>(work.initial_forcing->rhs_norms[block]));
            measure = std::max(measure, candidate_defects[block] / limit);
          }
          proof.measure = measure;
          proof.decision = measure <= 1 ? "eligible" : "rejected";
          // 完整gate才原子替换证明；初始快照和此前未完成候选均不能混入。
          *work.candidate_forcing = proof;
          ++work.candidate_forcing_checks;
          if (measure <= 1) {
            implementation::budget(check);
            work.final_measure = actual_assessment.measure;
            work.block_measures = actual_assessment.blocks;
            work.candidate_forcing->decision = "returned";
            work.status = "forced_candidate";
            ++work.candidate_forcing_returns;
            return std::move(actual_candidate);
          }
        }
        // 估计已过而真实仍未过时，仅在本次候选严格改善后用保留真残差重启。
        if (std::abs(transformed[column + 1]) <= 1 && candidate_improved) break;
        // 精确happy breakdown仍须真验；未通过时只可从保留真残差有限重启。
        if (remaining == 0) break;
      }
    }
  }
  implementation::budget(check);
  work.status = "exhausted";
  throw Failure("联立NewtonFGMRES未达到完整五块精度");
}

}  // 命名空间 solver::v100_coupled
